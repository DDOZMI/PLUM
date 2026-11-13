"""Deformable DETR: the PLUM dump of ID val, both OOD sets and the ID training reference images (one pass per image).

Per set one file bb5_<set>.pt with
  pair rows   : img_idx (global image index), query, cls (SIREN label order), score   -- every last-layer (query, class)
                pair with sigmoid > 0.1 (SIREN protocol)
  unique query: res5 (C5 7x7 mean at the clipped output box, fp16), box_net, uq_img, uq_query; row2uq (pair row -> query)
  per image   : input_hw (network input size); train also images = [(global index, image dict)]
Train = the train_subset reference images (VOC all, BDD the same 10k images as Faster R-CNN), in json order.
Output: $PLUM_DATA_ROOT/dumps/plum_ddetr_<task>/bb5_<set>.pt.

  python -m run.dump_ddetr --task voc
"""
import argparse

import torch

from plum import data as D
from plum import ddetr as M
from plum.config import dump_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["voc", "bdd"], required=True)
    ap.add_argument("--sets", nargs="*", default=None, help="subset of {<id val>, <ood sets>, <task>_train}; default all")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    D.register_all()
    task = args.task
    model, margs = M.build(task)
    outdir = dump_dir("ddetr", task)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"task": task, "protocol": f"SIREN: last-layer (query, class) pairs with sigmoid > {M.THR}",
            "map": "backbone[0].body.layer4 (C5, stride 32)", "box": "last-layer output box, network-input coords, clipped",
            "pool": "roi_align 7x7 aligned sampling 0, spatial mean, fp16", "checkpoint": str(M.CKPTS[task]),
            "torch": torch.__version__, "torchvision": __import__("torchvision").__version__, "PIL": __import__("PIL").__version__,
            "matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "transform": "SIREN val: RandomResize([800], max_size=1333) + ImageNet normalisation (RGB)",
            "img_idx": "position in the json image list (train: after removing the BDD unlabeled images)",
            "siren_args": {k: v for k, v in vars(margs).items() if isinstance(v, (int, float, str, bool))}}
    for name in [D.ID_VAL[task]] + D.OOD_SETS[task] + [f"{task}_train"]:
        if args.sets is not None and name not in args.sets:
            continue
        path = outdir / f"bb5_{name}.pt"
        if path.exists():
            print("skip", path); continue
        images = M.set_images(task, name)
        out = M.run(model, name, images)
        if name.endswith("_train"):
            out["images"] = [(g, {"id": im["id"], "file_name": im["file_name"], "height": im["height"], "width": im["width"]})
                             for g, im in images]
        out["meta"] = dict(meta, dataset=name, n_images=len(images))
        tmp = path.with_suffix(".pt.tmp"); torch.save(out, tmp); tmp.rename(path)
        print(f"saved {path}: {len(out['res5'])} unique queries, {len(out['row2uq'])} rows", flush=True)


if __name__ == "__main__":
    main()
