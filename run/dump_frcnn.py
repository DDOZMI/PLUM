"""Step 2 (Faster R-CNN): the PLUM dump of ID val, both OOD sets and the ID training reference images.

Per detection passing the VOS threshold (threshold.json of step 1): res5 7x7 mean at the proposal box (fp16), proposal box
(network coords), FPN level of the proposal, predicted class, det_score, proposal index; per image the resized size.
Train = data.train_subset (VOC all, BDD 10k seed 0), de-duplicated per proposal, in shards of 4000 images (resumable).
Output: $PLUM_DATA_ROOT/dumps/plum_frcnn_<task>/{<id val>.pt, <ood>.pt, <task>_train_shard###.pt, meta.json}.

  python -m run.dump_frcnn --task voc
"""
import argparse
import json
import time

import numpy as np
import torch

from plum import data as D
from plum import frcnn as F
from plum.config import dump_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["voc", "bdd"], required=True)
    ap.add_argument("--sets", nargs="*", default=None, help="subset of {<id val>, <ood sets>, <task>_train}; default all")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    D.register_all()
    task = args.task
    cfg, model = F.build(task)
    outdir = dump_dir("frcnn", task)
    thr = json.load(open(outdir / "threshold.json"))["threshold"]
    outdir.mkdir(parents=True, exist_ok=True)
    mp = outdir / "meta.json"
    runs = json.load(open(mp))["runs"] if mp.exists() else []
    F.write_json(mp, {**F.meta(task, cfg, thr), "runs": runs + [time.strftime("%Y-%m-%d %H:%M:%S")]})
    order = [(D.ID_VAL[task], False)] + [(o, False) for o in D.OOD_SETS[task]] + [(f"{task}_train", True)]
    for name, is_train in order:
        if args.sets is not None and name not in args.sets:
            continue
        path = outdir / f"{name}.pt"
        if not is_train and path.exists():
            print(f"skip existing {path}", flush=True)
            continue
        F.run(task, name, cfg, model, thr, is_train, outdir)
    print("DUMPDONE", flush=True)


if __name__ == "__main__":
    main()
