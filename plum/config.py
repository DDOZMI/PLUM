"""Paths. Everything is derived from two environment variables.

PLUM_DATA_ROOT   : datasets, checkpoints and dumps (default /mnt/d/datasets/od_ood, the layout of scripts/).
PLUM_THIRD_PARTY : directory holding the third-party code (default <repo>/third_party, i.e. the parent of this package's
                   folder + /third_party). Needed: vos/ (detectron2 configs + metric_utils.py) and siren/detr/ (patched).
                   detectron2 itself is an installed (editable) package, not imported from here.
PLUM_RESULTS     : where run/evaluate.py writes its tables (default <package>/results).
"""
import os
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PLUM_DATA_ROOT", "/mnt/d/datasets/od_ood"))
CKPT = DATA_ROOT / "checkpoints"
DUMP_ROOT = DATA_ROOT / "dumps"
THIRD_PARTY = Path(os.environ.get("PLUM_THIRD_PARTY", PKG.parent / "third_party"))
VOS_DIR = THIRD_PARTY / "vos"
SIREN_DETR_DIR = THIRD_PARTY / "siren" / "detr"
RESULTS = Path(os.environ.get("PLUM_RESULTS", PKG / "results"))


def dump_dir(det, task):
    """Dump directory of a detector x task, written by run/dump_*.py."""
    return DUMP_ROOT / f"plum_{det}_{task}"
