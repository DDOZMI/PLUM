"""Read the dumps into the inputs of PLUM (one dict per detector x task, same keys for both detectors).

  K        : number of ID classes
  sets     : ['id', <OOD set 1>, <OOD set 2>]
  ref_x    : res5 box features of the ID training reference rows (fp16, one row per reference detection)
  te_x     : {set: res5 box features of the test rows}; DDETR: one row per unique query, expanded by r2u
  r2u      : {set: pair row -> unique query index} (DDETR) or None (Faster R-CNN)
  cls_ref  : predicted class of each reference row
  split_idx: image index of each reference row (position in data.train_subset): even -> class statistics,
             odd with index % 4 == 1 -> D_L (lines), % 4 == 3 -> D_R (ranks)
  lvl_ref, L: FPN level (0..3) of the reference / test rows; cr, ct: log area and border contact (level.covariates)

Both readers take the dumps of run/dump_frcnn.py and run/dump_ddetr.py.
"""
import numpy as np
import torch
from detectron2.data import DatasetCatalog

from plum import data as D
from plum.level import covariates, pooler_level

TR_KEYS = ("img_idx", "det_class", "own_lvl", "proposal_box_net")


def row_hw(images, img_idx):
    return torch.tensor([images[i]["resized_hw"] for i in img_idx.tolist()], dtype=torch.float64)


def load_frcnn(task, ddir):
    K = D.NUM_CLASSES[task]
    names = {"id": D.ID_VAL[task], **{o: o for o in D.OOD_SETS[task]}}
    sets = list(names)
    T = {k: torch.load(ddir / f"{n}.pt", weights_only=False) for k, n in names.items()}
    parts, tr_images = {k: [] for k in TR_KEYS + ("res5",)}, []
    for p in sorted(ddir.glob(f"{task}_train_shard*.pt")):
        s = torch.load(p, weights_only=False)
        assert s["first_image"] == len(tr_images), p
        tr_images += s["images"]
        for k in TR_KEYS:
            parts[k].append(s[k])
        parts["res5"].append(s["res5"])
        del s
    D.register_all()
    # completeness: every training image and every test image of the sets is in the dumps
    assert len(tr_images) == len(D.train_subset(task)), (len(tr_images), task)
    for k, n in names.items():
        assert len(T[k]["images"]) == len(DatasetCatalog.get(n)), (k, n)
    tr = {k: torch.cat(v) for k, v in parts.items()}
    cr = covariates(tr["proposal_box_net"], row_hw(tr_images, tr["img_idx"]))
    ct = {k: covariates(T[k]["proposal_box_net"], row_hw(T[k]["images"], T[k]["img_idx"])) for k in sets}
    return dict(K=K, sets=sets, ref_x=tr["res5"], te_x={k: T[k]["res5"] for k in sets}, r2u={k: None for k in sets},
                cls_ref=tr["det_class"].long(), split_idx=tr["img_idx"].numpy(), lvl_ref=tr["own_lvl"].long().numpy(),
                cr=cr, ct=ct, L={k: T[k]["own_lvl"].long().numpy() for k in sets})


def load_ddetr(task, ddir):
    K = D.NUM_CLASSES[task]
    names = {"id": D.ID_VAL[task], **{o: o for o in D.OOD_SETS[task]}}
    sets = list(names)
    B = {k: torch.load(ddir / f"bb5_{n}.pt", map_location="cpu", weights_only=False) for k, n in names.items()}
    Tr = torch.load(ddir / f"bb5_{task}_train.pt", map_location="cpu", weights_only=False)
    images, trows = Tr["images"], {k: Tr[k] for k in ("img_idx", "query", "cls", "score")}
    sel = np.array([g for g, _ in images])
    # class of a unique train query = class of its highest-score pair
    u = Tr["row2uq"].numpy()
    order = np.lexsort((np.arange(len(u)), -trows["score"].double().numpy(), u))
    first = order[np.unique(u[order], return_index=True)[1]]
    ctr = torch.from_numpy(trows["cls"].numpy()[first].astype(np.int64))
    assert len(ctr) == len(Tr["res5"]) and np.array_equal(u[first], np.arange(len(ctr)))
    # image index of each reference row = position of its image in data.train_subset (the Faster R-CNN reference order)
    D.register_all()
    refl = D.train_subset(task)
    # completeness: every training image and every test image of the sets is in the dumps
    assert len(images) == len(refl), (len(images), task)
    for k, n in names.items():
        assert len(B[k]["input_hw"]) == len(DatasetCatalog.get(n)), (k, n)
    pos_of = {x["file_name"].split("/")[-1]: i for i, x in enumerate(refl)}
    fname = {g: im["file_name"] for g, im in images}
    gi = Tr["uq_img"].numpy()
    pos = np.array([pos_of[fname[int(g)]] for g in gi])
    hw_tr = Tr["input_hw"][torch.from_numpy(np.searchsorted(sel, gi))]
    cr = covariates(Tr["box_net"], hw_tr)
    r2u, ct, L = {}, {}, {}
    for k in sets:
        r = B[k]["row2uq"]
        r2u[k] = r.numpy()
        box = B[k]["box_net"][r]
        ct[k] = covariates(box, B[k]["input_hw"][B[k]["uq_img"][r].long()])
        L[k] = pooler_level(box)
    return dict(K=K, sets=sets, ref_x=Tr["res5"], te_x={k: B[k]["res5"] for k in sets}, r2u=r2u, cls_ref=ctr, split_idx=pos,
                lvl_ref=pooler_level(Tr["box_net"]), cr=cr, ct=ct, L=L)
