"""Derive BDD/train_bdd_converted_fixed.json from the VOS json (original untouched).

Defect: 10 image ids are each shared by two consecutive images of the VOS BDD train json. The bdd100k_id continuity shows
all annotations of a shared id belong to the SECOND image; the FIRST image had zero box labels in the source label version.
Fix: give each first image a new unique id (max_id+1, ...) with no annotations. Annotations are not modified.
Also writes BDD/train_unlabeled_image_ids.json listing the reassigned images; PLUM excludes them from the ID training
reference images (plum.data.train_subset).
"""
import collections
import json
import os
from pathlib import Path

BDD = Path(os.environ.get("PLUM_DATA_ROOT", "/mnt/d/datasets/od_ood")) / "BDD"
src = BDD / "train_bdd_converted.json"
d = json.load(open(src))

by_id = collections.defaultdict(list)
for k, im in enumerate(d["images"]):
    by_id[im["id"]].append(k)
dups = {i: ks for i, ks in by_id.items() if len(ks) > 1}
assert len(dups) == 10 and all(len(ks) == 2 and ks[1] == ks[0] + 1 for ks in dups.values()), dups

next_id = max(by_id) + 1
reassigned = []
for old_id in sorted(dups):
    first = d["images"][dups[old_id][0]]
    first["id"] = next_id
    reassigned.append({"file_name": first["file_name"], "old_id": old_id, "new_id": next_id})
    next_id += 1

assert len(d["annotations"]) == 1273707  # annotations untouched
ids = [im["id"] for im in d["images"]]
assert len(ids) == len(set(ids)) == 69863
ann_ids = {a["image_id"] for a in d["annotations"]}
assert all(i in ann_ids for i in dups)  # old ids keep their annotations (now the second image only)

json.dump(d, open(BDD / "train_bdd_converted_fixed.json", "w"))
json.dump({"reason": "first image of each duplicate-id pair; no box labels in source label version",
           "images": reassigned}, open(BDD / "train_unlabeled_image_ids.json", "w"), indent=1)
print(json.dumps(reassigned, indent=1))
