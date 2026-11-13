"""Faster R-CNN R50-FPN (VOS vanilla checkpoints): model, VOS-protocol inference, the VOS F1 threshold and the PLUM dump.

Inference follows the VOS/ProbDet 'standard_nms' protocol (third_party/vos/detection/inference/rcnn_predictor.py):
  1. backbone -> RPN proposals (post-NMS top-1000) -> RoIAlign on proposals -> box head -> predictor.
  2. softmax over 21 (VOC) / 11 (BDD) logits; foreground scores = softmax[:, :-1].
  3. keep (proposal, class) pairs with fg score > SCORE_THRESH_TEST (0.05); apply class-specific deltas
     (boxes are NOT clipped before NMS, as in VOS).
  4. batched_nms per class at NMS_THRESH_TEST (0.5); keep first DETECTIONS_PER_IMAGE (100).
  5. rescale boxes to the original image size, clip, drop empty boxes.
Detection threshold (VOS protocol, Sec. 5 of the paper): f1_threshold on ID val; a detection is kept iff the max foreground
softmax of its proposal is >= the threshold, for ID and OOD alike (and for the ID training reference images).

PLUM dump (paper Sec. 4.1, Eq. pool): per kept detection, the res5 output (bottom-up stage res5, stride 32, 2048 channels)
RoIAligned on a 7x7 grid (aligned, sampling_ratio 0, as the box pooler) at the PROPOSAL box in network-input coordinates (the
box the box head pools), averaged over the 49 cells, stored fp16; plus the proposal box, the pooler's FPN level of the
proposal (= Eq. level, 0..3 for P2..P5), the predicted class and the image sizes. Training reference images are de-duplicated
per proposal (a proposal kept for several classes is one reference row: the one with the highest det_score).
"""
import contextlib
import hashlib
import io
import json
import time

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.layers import batched_nms
from detectron2.modeling import build_model
from detectron2.modeling.poolers import assign_boxes_to_levels
from detectron2.structures import Boxes
from torchvision.ops import roi_align

from plum import data as D
from plum.config import CKPT, VOS_DIR

VOS_CFG = VOS_DIR / "detection/configs"
CKPTS = {"voc": CKPT / "vos/model_final_vanilla_resnet_voc.pth", "bdd": CKPT / "vos/model_final_vanilla_resnet_bdd.pth"}
RES5_STRIDE = 32
SHARD = 4000
FIELDS = ["img_idx", "prop_index", "det_class", "det_score", "own_lvl", "proposal_box_net", "res5"]


def build(task):
    cfg = get_cfg()
    cfg.merge_from_file(str(VOS_CFG / ("VOC-Detection" if task == "voc" else "BDD-Detection") / "faster-rcnn/vanilla.yaml"))
    cfg.MODEL.WEIGHTS = str(CKPTS[task])
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.freeze()
    model = build_model(cfg).eval()
    ckpt = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu", weights_only=False)["model"]
    missing = [k for k in model.state_dict() if k not in ckpt]
    assert not missing, f"checkpoint missing model keys: {missing[:10]}"
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return cfg, model


def run_head(model, fl, boxes_list):
    rh = model.roi_heads
    x = torch.flatten(rh.box_pooler(fl, boxes_list), 1)
    head = rh.box_head
    fc1 = head.fc_relu1(head.fc1(x))
    fc2 = head.fc_relu2(head.fc2(fc1))
    logits, deltas = rh.box_predictor(fc2)
    return fc1, fc2, logits, deltas


@torch.no_grad()
def infer_one(model, inp):
    """VOS standard_nms inference for one image. Returns (out, (resized h, w), internal); internal holds all proposals'
    boxes (network coords) and each output detection's proposal index."""
    rh = model.roi_heads
    pred = rh.box_predictor
    images = model.preprocess_image([inp])
    feats = model.backbone(images.tensor)
    proposals, _ = model.proposal_generator(images, feats, None)
    prop = proposals[0]
    fl = [feats[f] for f in rh.box_in_features]
    fc1, fc2, logits, deltas = run_head(model, fl, [prop.proposal_boxes])

    probs = torch.softmax(logits, -1)
    scores = probs[:, :-1]
    K = scores.shape[1]
    deltas = deltas.view(-1, deltas.shape[1] // 4, 4)
    mask = scores > pred.test_score_thresh
    inds = mask.nonzero(as_tuple=False)  # (proposal, class)
    pidx, cls = inds[:, 0], inds[:, 1]
    d = deltas[mask] if deltas.shape[1] > 1 else deltas[pidx, 0]
    pboxes = prop.proposal_boxes.tensor[pidx]
    boxes = pred.box2box_transform.apply_deltas(d, pboxes)
    sc = scores[mask]
    keep = batched_nms(boxes, sc, cls, pred.test_nms_thresh)[: pred.test_topk_per_image]
    pidx, cls, boxes, sc, d, pboxes = pidx[keep], cls[keep], boxes[keep], sc[keep], d[keep], pboxes[keep]

    # rescale to original resolution, clip, drop empty (probabilistic_detector_postprocess)
    rh_, rw_ = images.image_sizes[0]
    H, W = inp["height"], inp["width"]
    sx, sy = W / rw_, H / rh_
    b = Boxes(boxes.clone()); b.scale(sx, sy); b.clip((H, W))
    ne = b.nonempty()
    out = {
        "prop_index": pidx[ne].int(), "det_class": cls[ne].short(), "det_score": sc[ne].float(),
        "box": b.tensor[ne].float(), "logits": logits[pidx[ne]].float(),
    }
    assert K == logits.shape[1] - 1
    internal = {"proposal_boxes": prop.proposal_boxes.tensor, "pidx": pidx[ne]}
    return out, (rh_, rw_), internal


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ VOS F1 threshold
@torch.no_grad()
def id_val_detections(task, cfg, model):
    """All VOS-protocol detections on ID val (no threshold), with the fields f1_threshold reads."""
    name = D.ID_VAL[task]
    loader = build_detection_test_loader(cfg, dataset_name=name)
    fields, images = {}, []
    t0 = time.time()
    for n, batch in enumerate(loader):
        inp = batch[0]
        out, rsz, _ = infer_one(model, inp)
        m = len(out["det_score"])
        out["img_idx"] = torch.full((m,), len(images), dtype=torch.int32)  # index into images
        for key in ("img_idx", "det_class", "det_score", "box"):
            fields.setdefault(key, []).append(out[key].cpu())
        images.append({"image_id": inp["image_id"], "file_name": inp["file_name"].split("/")[-1],
                       "height": inp["height"], "width": inp["width"], "resized_hw": list(rsz)})
        if n % 500 == 0:
            print(f"{name} {n}/{len(loader)} {time.time() - t0:.0f}s", flush=True)
    dump = {key: torch.cat(v) for key, v in fields.items()}
    dump["images"] = images
    return dump


def f1_threshold(dump, gt_json):
    """VOS offline_evaluation/compute_average_precision.py: COCOeval on ID val; per-class score at the F1-optimal recall point
    (precision averaged over IoU thresholds, area=all, maxDets=100), zeros removed, averaged over classes."""
    res = []
    iids = [im["image_id"] for im in dump["images"]]
    for ii, c, s, b in zip(dump["img_idx"].tolist(), dump["det_class"].tolist(), dump["det_score"].tolist(),
                           dump["box"].tolist()):
        iid = iids[ii]
        res.append({"image_id": iid, "category_id": c + 1, "score": s,
                    "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_json))
        dt = gt.loadRes(res)
        E = COCOeval(gt, dt, iouType="bbox")
        E.params.catIds = sorted(gt.getCatIds())
        E.evaluate(); E.accumulate(); E.summarize()
    precisions = E.eval["precision"].mean(0)[:, :, 0, 2]
    recalls = np.expand_dims(E.params.recThrs, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2 * (precisions * recalls) / (precisions + recalls)
    opt = f1.argmax(0)
    scores = E.eval["scores"].mean(0)[:, :, 0, 2]
    thr = np.array([scores[o, i] for i, o in enumerate(opt)])
    thr = thr[thr != 0].mean()
    return float(thr), E.stats.tolist()


# ------------------------------------------------------------------------------------------------ PLUM dump
def pool_mean(f, boxes, stride):
    return roi_align(f, [boxes], (7, 7), 1.0 / stride, sampling_ratio=0, aligned=True).mean((2, 3))


@torch.no_grad()
def run(task, name, cfg, model, thr, is_train, outdir):
    """One set. Test sets -> <outdir>/<name>.pt; train (the train_subset reference images) -> <outdir>/<name>_shard###.pt
    (SHARD images per shard, resumable). img_idx = index into the file's images (test) / GLOBAL loader index (train shards;
    the images list of a shard starts at first_image)."""
    dicts = D.train_subset(task) if is_train else None
    loader = (build_detection_test_loader(dicts, mapper=DatasetMapper(cfg, is_train=False)) if is_train
              else build_detection_test_loader(cfg, dataset_name=name))
    st = {}
    hook = model.backbone.bottom_up.res5.register_forward_hook(lambda m, i, o: st.__setitem__("res5", o))
    pooler = model.roi_heads.box_pooler
    fields, images = {k: [] for k in FIELDS}, []
    t0 = time.time()

    def flush(k_shard, imgs):
        d = {k: torch.cat(v) if v else torch.empty(0) for k, v in fields.items()}
        d.update(images=imgs, first_image=k_shard * SHARD if is_train else 0)
        p = outdir / (f"{name}_shard{k_shard:03d}.pt" if is_train else f"{name}.pt")
        tmp = p.with_suffix(".pt.tmp"); torch.save(d, tmp); tmp.rename(p)
        print(f"saved {p}: {len(imgs)} images, {len(d['det_score'])} rows", flush=True)
        return d

    n_iter = 0
    for n, batch in enumerate(loader):
        n_iter = n + 1
        if is_train and (outdir / f"{name}_shard{n // SHARD:03d}.pt").exists():
            if n % SHARD == 0:
                print(f"skip existing shard {n // SHARD}", flush=True)
            continue
        inp = batch[0]
        out, (rh, rw), internal = infer_one(model, inp)
        keep = torch.softmax(out["logits"], -1)[:, :-1].max(1).values >= thr
        idx = torch.nonzero(keep).flatten()
        if is_train and len(idx):  # per-proposal de-duplication of the reference rows
            key = out["prop_index"][idx].long().cpu()
            order = torch.argsort(out["det_score"][idx].cpu(), descending=True)
            _, first = np.unique(key[order].numpy(), return_index=True)
            idx = idx[order[torch.from_numpy(first)].to(idx.device)].sort().values
        m = len(idx)
        pidx = internal["pidx"][idx].long()
        pbox = internal["proposal_boxes"][pidx]  # network coords
        if m:
            own = assign_boxes_to_levels([Boxes(pbox)], pooler.min_level, pooler.max_level, pooler.canonical_box_size,
                                         pooler.canonical_level)
            fields["own_lvl"].append(own.to(torch.int8).cpu())
            fields["res5"].append(pool_mean(st["res5"], pbox, RES5_STRIDE).half().cpu())
            fields["proposal_box_net"].append(pbox.float().cpu())
            fields["img_idx"].append(torch.full((m,), n, dtype=torch.int32))
            fields["prop_index"].append(out["prop_index"][idx].cpu())
            fields["det_class"].append(out["det_class"][idx].cpu())
            fields["det_score"].append(out["det_score"][idx].cpu())
        images.append({"image_id": inp["image_id"], "file_name": inp["file_name"].split("/")[-1], "height": inp["height"],
                       "width": inp["width"], "resized_hw": [rh, rw]})
        if n % 1000 == 0:
            print(f"{name} {n}/{len(loader)} {time.time() - t0:.0f}s", flush=True)
        if is_train and (n + 1) % SHARD == 0:
            flush(n // SHARD, images)
            fields, images = {k: [] for k in FIELDS}, []
    hook.remove()
    if is_train:
        if images:  # last partial shard
            flush(n_iter // SHARD, images)
        return None
    return flush(0, images)


def meta(task, cfg, thr):
    return {"task": task, "threshold": thr, "checkpoint": str(cfg.MODEL.WEIGHTS), "checkpoint_sha256": sha256(cfg.MODEL.WEIGHTS),
            "torch": torch.__version__, "detectron2": __import__("detectron2").__version__,
            "cudnn_conv_fp32_precision": torch.backends.cudnn.conv.fp32_precision,
            "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision, "fields": FIELDS, "shard": SHARD,
            "dataset_json_sha256": {n: sha256(D.DATASETS[n][0]) for n in [D.ID_VAL[task], f"{task}_train"] + D.OOD_SETS[task]},
            "cfg": cfg.dump()}


def write_json(path, obj):
    json.dump(obj, open(path, "w"), indent=1, default=str)
