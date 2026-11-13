"""Deformable DETR (SIREN vanilla checkpoints): model, SIREN-protocol detections and the PLUM dump, in one pass per image.

Model: third_party/siren/detr (SIREN commit 3bb2a6a6 + third_party/siren_torch2_{detr,msda}.patch) built with the checkpoint's
stored args (defaults from get_args_parser for anything missing), load_backbone='none' (no network; the checkpoint overwrites
the backbone), SIREN head disabled (vanilla); strict state_dict loading (asserted). Preprocessing = SIREN 'val' transform
(RandomResize([800], max_size=1333) + ImageNet normalisation, RGB), batch 1 (no padding).

Detections (SIREN protocol, deformable_detr.py PostProcess): every (query, class) pair of the LAST decoder layer with
sigmoid(logit) > 0.1; no NMS, no top-k; a query passing for several classes appears once per class (one row per pair).

PLUM dump (paper Sec. 4.1): the box from which DDETR reads the object is the query's last-layer output box. Per unique
(image, query): the backbone C5 output (backbone[0].body.layer4, ResNet-50 res5, stride 32, 2048 channels) RoIAligned 7x7
(aligned, sampling_ratio 0) at the output box in network-input coordinates, clipped to the image, averaged over the 49
cells, stored fp16, with the clipped box and the input size. row2uq maps every pair row to its unique query.
Training reference rows (loaded in plum.load): one row per unique query, class = class of its highest-score pair.
"""
import importlib.util
import json
import sys
import time

import torch
from PIL import Image
from torchvision.ops import roi_align

from plum import data as D
from plum.config import CKPT, SIREN_DETR_DIR

sys.path.insert(0, str(SIREN_DETR_DIR))
_spec = importlib.util.spec_from_file_location("siren_transforms", SIREN_DETR_DIR / "datasets/transforms.py")
T = importlib.util.module_from_spec(_spec)  # SIREN's datasets/transforms.py, loaded by path (avoids the 'datasets' name)
_spec.loader.exec_module(T)
from models import build_model  # noqa: E402
from util.default_args import get_args_parser, set_model_defaults  # noqa: E402
from util.misc import nested_tensor_from_tensor_list  # noqa: E402

CKPTS = {t: CKPT / "siren" / f"checkpoint_{t}_vanilla.pth" for t in ("voc", "bdd")}
THR = 0.1
STRIDE = 32

_TF = T.Compose([T.RandomResize([800], max_size=1333),
                 T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])])


def build(task):
    ck = torch.load(CKPTS[task], map_location="cpu", weights_only=False)
    args = get_args_parser().parse_args([])
    for k, v in vars(ck["args"]).items():
        setattr(args, k, v)
    args.device, args.load_backbone, args.dataset_file, args.dataset = "cuda", "none", task, task
    args.siren = False
    args = set_model_defaults(args)
    model, _, _ = build_model(args)
    res = model.load_state_dict(ck["model"], strict=False)
    assert not res.missing_keys and not res.unexpected_keys, (res.missing_keys[:5], res.unexpected_keys[:5])
    return model.cuda().eval(), args


def preprocess(pil):
    img, _ = _TF(pil.convert("RGB"), None)
    return img


@torch.no_grad()
def forward(model, img):
    """one image -> last decoder layer logits (Q, C) and boxes (Q, 4; normalised cxcywh)."""
    samples = nested_tensor_from_tensor_list([img.cuda()])
    out = model(samples)  # forward() sets model.epoch (unused at eval) then calls forward_samples
    return {"logits": out["pred_logits"][0], "boxes": out["pred_boxes"][0]}


def load_image(dataset, file_name):
    return Image.open(D.DATASETS[dataset][1] / file_name)


def set_images(task, name):
    """(global index, image dict) pairs of a set, in json order. Global index = position in the json image list (train: after
    removing the BDD unlabeled images). Train keeps only the train_subset reference images (VOC all;
    BDD the 10k images of the Faster R-CNN reference, matched by file name)."""
    js = json.load(open(D.DATASETS[name][0]))
    if not name.endswith("_train"):
        return list(enumerate(js["images"]))
    excluded = set() if task == "voc" else {x["file_name"] for x in json.load(open(D.BDD_TRAIN_UNLABELED))["images"]}
    images = list(enumerate(im for im in js["images"] if im["file_name"] not in excluded))
    if task == "voc":
        return images
    want = {d["file_name"].split("/")[-1] for d in D.train_subset(task)}
    images = [(i, im) for i, im in images if im["file_name"] in want]
    assert len(images) == len(want) == D.BDD_TRAIN_N, (len(images), len(want))
    return images


@torch.no_grad()
def run(model, name, images):
    """images: list of (global_idx, image dict). Returns pair rows (img_idx, query, cls, score) and unique-query features."""
    feat = {}
    h = model.backbone[0].body.layer4.register_forward_hook(lambda m, i, o: feat.__setitem__("c5", o))
    rows = {k: [] for k in ("img_idx", "query", "cls", "score")}
    res5, box_net, uq_img, uq_query, row2uq, in_hw = [], [], [], [], [], []
    nu, t0 = 0, time.time()
    for n, (gi, im) in enumerate(images):
        pil = load_image(name, im["file_name"])
        x = preprocess(pil)
        hi, wi = x.shape[1:]
        in_hw.append((hi, wi))
        r = forward(model, x)
        prob = r["logits"].sigmoid()  # (Q, C)
        q, c = torch.nonzero(prob > THR, as_tuple=True)
        k = len(q)
        if k == 0:
            continue
        rows["img_idx"].append(torch.full((k,), gi, dtype=torch.int32))
        rows["query"].append(q.short().cpu()); rows["cls"].append(c.short().cpu())
        rows["score"].append(prob[q, c].float().cpu())
        b = r["boxes"]
        uq, inv = torch.unique(q, return_inverse=True)
        cx, cy, bw, bh = b[uq].unbind(-1)
        bn = torch.stack([(cx - bw / 2) * wi, (cy - bh / 2) * hi, (cx + bw / 2) * wi, (cy + bh / 2) * hi], -1)
        bn = torch.stack([bn[:, 0].clamp(0, wi), bn[:, 1].clamp(0, hi), bn[:, 2].clamp(0, wi), bn[:, 3].clamp(0, hi)], -1)
        res5.append(roi_align(feat["c5"], [bn], (7, 7), 1.0 / STRIDE, sampling_ratio=0, aligned=True).mean((2, 3)).half().cpu())
        box_net.append(bn.float().cpu())
        uq_img.append(torch.full((len(uq),), gi, dtype=torch.int32)); uq_query.append(uq.short().cpu())
        row2uq.append((inv + nu).cpu())
        nu += len(uq)
        if n % 2000 == 0:
            print(f"{name} {n}/{len(images)} {time.time() - t0:.0f}s", flush=True)
    h.remove()
    out = {k: torch.cat(v) for k, v in rows.items()}
    out.update({"res5": torch.cat(res5), "box_net": torch.cat(box_net), "uq_img": torch.cat(uq_img),
                "uq_query": torch.cat(uq_query), "row2uq": torch.cat(row2uq), "input_hw": torch.tensor(in_hw)})
    return out

