"""Inspect PUBLIC COCO annotations without downloading the full dataset."""
from huggingface_hub import hf_hub_download
import json
from collections import Counter
repo="imaadd05/paint-defects-roboflow-main"
for split in ("train","valid","test"):
    file=hf_hub_download(repo_id=repo,repo_type="dataset",filename=split+"/_annotations.coco.json")
    data=json.load(open(file,encoding="utf-8"))
    names={str(v["id"]):v["name"] for v in data["categories"]}
    frequencies=Counter(names.get(str(a.get("category_id")),"unknown") for a in data["annotations"])
    print(f"PUBLIC DATASET AUDIT {split}:",json.dumps({"images":len(data["images"]),"classes":names,"instances":dict(frequencies)},ensure_ascii=False))
