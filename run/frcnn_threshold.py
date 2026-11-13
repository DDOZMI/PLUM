"""Step 1 (Faster R-CNN): the VOS detection threshold of a task.

All VOS-protocol detections on ID val -> COCOeval -> mean over classes of the F1-optimal score (plum.frcnn.f1_threshold),
rounded to 4 decimals (VOS compute_ood_probabilistic_metrics.py). Paper values: VOC 0.5728, BDD 0.4006.
Output: $PLUM_DATA_ROOT/dumps/plum_frcnn_<task>/threshold.json.

  python -m run.frcnn_threshold --task voc
"""
import argparse

import numpy as np
import torch

from plum import data as D
from plum import frcnn as F
from plum.config import dump_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["voc", "bdd"], required=True)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.conv.fp32_precision = "ieee"  # VOS ran true FP32; torch 2.x would use TF32 convs otherwise
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    D.register_all()
    cfg, model = F.build(args.task)
    outdir = dump_dir("frcnn", args.task)
    outdir.mkdir(parents=True, exist_ok=True)
    dump = F.id_val_detections(args.task, cfg, model)
    idname = D.ID_VAL[args.task]
    thr_raw, ap_stats = F.f1_threshold(dump, D.DATASETS[idname][0])
    thr = round(thr_raw, 4)
    print(f"[{args.task}] ID AP stats (AP, AP50, AP75): {[round(x, 4) for x in ap_stats[:3]]}  F1-opt threshold {thr_raw:.6f} -> {thr}")
    F.write_json(outdir / "threshold.json", {"task": args.task, "threshold": thr, "threshold_raw": thr_raw, "id_ap_stats": ap_stats,
                                 "n_images": len(dump["images"]), "n_dets": int(len(dump["det_score"])),
                                 "meta": F.meta(args.task, cfg, None)})


if __name__ == "__main__":
    main()
