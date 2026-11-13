"""Step 3: PLUM AUROC and FPR95 (ID = positive) per OOD set for each detector x task.

  python -m run.evaluate                                   # all four detector x task pairs, package dumps
  python -m run.evaluate --det frcnn --task voc

Prints the table and writes $PLUM_RESULTS/plum_<det>_<task>.csv (columns 'AUROC <ood>', 'FPR95 <ood>') and
u_<det>_<task>.pt ({set: u, stratum, raw score} per test row, float64).
Expected (paper Table 1, Faster R-CNN): VOC COCO 91.46 / 44.16, OI 94.50 / 29.25; BDD COCO 95.07 / 24.35, OI 97.51 / 13.87.
"""
import argparse

import numpy as np
import pandas as pd
import torch

from plum.config import RESULTS, dump_dir
from plum.load import load_ddetr, load_frcnn
from plum.method import plum_values
from plum.metrics import measures


def evaluate(det, task):
    ddir = dump_dir(det, task)
    X = (load_frcnn if det == "frcnn" else load_ddetr)(task, ddir)
    U, C, S = plum_values(X)
    sets = X["sets"]
    row = {"det": det, "task": task, "dump_dir": str(ddir), "n_id": len(U["id"])}
    for o in sets[1:]:
        row[f"AUROC {o}"], row[f"FPR95 {o}"] = measures(U["id"], U[o])
        row[f"n {o}"] = len(U[o])
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"{det}_{task}"
    pd.DataFrame([row]).to_csv(RESULTS / f"plum_{tag}.csv", index=False)
    torch.save({k: {"u": torch.from_numpy(np.asarray(U[k], np.float64)), "stratum": torch.from_numpy(np.asarray(C[k])),
                    "raw": torch.from_numpy(np.asarray(S[k], np.float64))} for k in sets}, RESULTS / f"u_{tag}.pt")
    del X
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", choices=["frcnn", "ddetr", "all"], default="all")
    ap.add_argument("--task", choices=["voc", "bdd", "all"], default="all")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    dets = ["frcnn", "ddetr"] if a.det == "all" else [a.det]
    tasks = ["voc", "bdd"] if a.task == "all" else [a.task]
    for det in dets:
        for task in tasks:
            row = evaluate(det, task)
            print(f"\nPLUM | {det} | ID {task} | {row['dump_dir']}")
            for k in [c for c in row if c.startswith(("AUROC", "FPR95"))]:
                print(f"  {k:24s} {row[k]:.2f}")


if __name__ == "__main__":
    main()
