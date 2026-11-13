"""Datasets of the VOS / SIREN OOD object detection benchmark and the ID training reference images.

Category mapping follows VOS metadata: json category id k -> contiguous (network output) index k-1, for both VOC (20) and
BDD (10). OOD sets carry no ID categories; their category ids are irrelevant.

Paper (Sec. 4.2): PLUM runs the detector on the ID training images. VOC uses all 16,551 training images. BDD uses 10,000
training images drawn with seed 0 (train_subset), after removing the 10 images that lost their labels in the VOS json
(scripts/fix_bdd_train_ids.py).
"""
import json
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from plum.config import DATA_ROOT as ROOT

VOC_CLASSES = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "airplane", "bicycle", "boat", "bus",
               "car", "motorcycle", "train", "bottle", "chair", "dining table", "potted plant", "couch", "tv"]
BDD_CLASSES = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
               "traffic light", "traffic sign"]

# name -> (json, image dir, ID task it belongs to or is OOD for)
DATASETS = {
    "voc_train": (ROOT / "VOC/VOC_0712_converted/voc0712_train_all.json", ROOT / "VOC/VOC_0712_converted/JPEGImages", "voc"),
    "voc_val": (ROOT / "VOC/VOC_0712_converted/val_coco_format.json", ROOT / "VOC/VOC_0712_converted/JPEGImages", "voc"),
    "coco_ood_voc": (ROOT / "COCO/annotations/instances_val2017_ood_rm_overlap.json", ROOT / "COCO/val2017", "voc"),
    "oi_ood_voc": (ROOT / "OpenImages/ood_classes_rm_overlap/COCO-Format/val_coco_format.json",
                   ROOT / "OpenImages/ood_classes_rm_overlap/images", "voc"),
    # BDD train: id-fixed json (scripts/fix_bdd_train_ids.py); BDD_TRAIN_UNLABELED is excluded from the ID statistics.
    "bdd_train": (ROOT / "BDD/train_bdd_converted_fixed.json", ROOT / "BDD/images/100k/train", "bdd"),
    "bdd_val": (ROOT / "BDD/val_bdd_converted.json", ROOT / "BDD/images/100k/val", "bdd"),
    "coco_ood_bdd": (ROOT / "COCO/annotations/instances_val2017_ood_wrt_bdd_rm_overlap.json", ROOT / "COCO/val2017", "bdd"),
    "oi_ood_bdd": (ROOT / "OpenImages/ood_classes_rm_overlap/COCO-Format/val_coco_format.json",
                   ROOT / "OpenImages/ood_classes_rm_overlap/images", "bdd"),
}
BDD_TRAIN_UNLABELED = ROOT / "BDD/train_unlabeled_image_ids.json"
ID_VAL = {"voc": "voc_val", "bdd": "bdd_val"}
OOD_SETS = {"voc": ["coco_ood_voc", "oi_ood_voc"], "bdd": ["coco_ood_bdd", "oi_ood_bdd"]}
NUM_CLASSES = {"voc": 20, "bdd": 10}

BDD_TRAIN_N = 10000
# file names of the BDD training subset of the paper, written once from the original selection
# (np.random.default_rng(0).choice over the fixed json minus the unlabeled images, sorted); stored so that the subset
# does not depend on the random stream of a numpy version
BDD_TRAIN_LIST = Path(__file__).with_name("bdd_train_10k.txt")


def register_all():
    for name, (js, imdir, task) in DATASETS.items():
        if name in DatasetCatalog.list():
            continue
        register_coco_instances(name, {}, str(js), str(imdir))
        if not name.startswith(("coco_ood", "oi_ood")):
            classes = VOC_CLASSES if task == "voc" else BDD_CLASSES
            md = MetadataCatalog.get(name)
            md.thing_classes = classes
            md.thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(classes))}


def train_subset(task):
    """VOC: all train images. BDD: the 10k images of BDD_TRAIN_LIST (drawn with seed 0 from the fixed json minus the 10
    unlabeled images).
    Order = detectron2 DatasetCatalog order (load_coco_json sorts by image id); the position in this list is the image
    index whose parity / residue mod 4 defines the PLUM split (even = class statistics, %4 == 1 = D_L, %4 == 3 = D_R)."""
    dicts = DatasetCatalog.get(f"{task}_train")
    if task == "voc":
        return dicts
    unl = {x["file_name"] for x in json.load(open(BDD_TRAIN_UNLABELED))["images"]}
    want = set(BDD_TRAIN_LIST.read_text().split())
    sub = [d for d in dicts if d["file_name"].split("/")[-1] not in unl and d["file_name"].split("/")[-1] in want]
    assert len(sub) == len(want) == BDD_TRAIN_N, (len(sub), len(want))
    return sub
