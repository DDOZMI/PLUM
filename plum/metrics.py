"""AUROC and FPR95 with ID as the positive class, via the VOS metric code (third_party/vos/metric_utils.get_measures:
sklearn roc_auc_score and the FPR at 95 % TPR). Inputs are cast to float64. Returned in percent."""
import sys

import numpy as np

from plum.config import VOS_DIR

sys.path.insert(0, str(VOS_DIR))
from metric_utils import get_measures  # noqa: E402


def measures(sid, so):
    au, _, fp = get_measures(np.asarray(sid, np.float64), np.asarray(so, np.float64))
    return 100 * au, 100 * fp
