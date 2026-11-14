"""Download and fine-tune a YOLO candidate on REAL CC-BY annotated paint images.

Public dataset: imaadd05/paint-defects-roboflow-main (Hugging Face).
Only COCO categories paint_defect are mapped, with a held-out TEST split.
No training with assistant-generated screenshots, no invented labels.
The training output is a candidate and is NEVER automatically deployed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import shutil

SOURCE="imaadd05/paint-defects-roboflow-main"
SOURCE_LICENSE="CC BY 4.0"
SOURCE_URL="https://huggingface.co/datasets/"+SOURCE
TARGET_CLASS="pintura_deteriorada"
ACCEPTED_SOURCE_NAMES={"paint_defect","paint defect","paint defects"}

def category_name(name):
    return re.sub(r"[^a-z0-9]+","_",str(name).strip().lower()).strip("_")

def convert_coco(root:Path,destination:Path,min_per_split=5,max_train=450,seed=42):
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("El destino ya existe: no se mezclan ejecuciones")
    rng=random.Random(seed)
    report={"source":SOURCE_URL,"license":SOURCE_LICENSE,"class":TARGET_CLASS,
            "splits":{},"image_hashes_by_split":{},"status":"dataset_preparado"}
    seen_hashes=set()
    family_splits={}
    all_records={}
    for split in ("train","valid","test"):
        folder=root/split
        ann=folder/"_annotations.coco.json"
        if not ann.is_file():
            raise FileNotFoundError("Falta anotación COCO verificable: "+str(ann))
        data=json.loads(ann.read_text(encoding="utf-8"))
        mapping={c["id"]:category_name(c["name"]) for c in data["categories"]}
        targets={i for i,n in mapping.items() if n in {category_name(x) for x in ACCEPTED_SOURCE_NAMES}}
        if not targets:raise ValueError("El dataset no incluye paint_defect en "+split+": "+str(mapping))
        items={}
        for obj in data["annotations"]:
            if obj.get("category_id") not in targets or obj.get("iscrowd",0):continue
            items.setdefault(obj["image_id"],[]).append(obj)
        info={i["id"]:i for i in data["images"]}
        selected=[]
        for idx,annotations in items.items():
            item=info.get(idx)
            if item is None:raise ValueError("Anotación sin imagen")
            name=Path(item["file_name"]).name
            src=folder/name
            if not src.is_file():raise FileNotFoundError(str(src))
            if src.suffix.lower() not in (".jpg",".jpeg",".png",".webp"):continue
            # Roboflow names include augmentation hashes; family is the prefix.
            family=re.sub(r"\.(?:jpg|png|jpeg|webp)\.rf\.[0-9a-f]+$","",src.stem,flags=re.I)
            selected.append((src,item,annotations,family))
        if split=="train" and max_train>0 and len(selected)>max_train:
            rng.shuffle(selected)
            selected=selected[:max_train]
        if len(selected)<min_per_split:
            raise ValueError(f"{split}: {len(selected)} positivas, mínimo {min_per_split}")
        all_records[split]=selected
        report["splits"][split]={"source_positives":len(items),"selected":len(selected)}
    # Grouped leak check by image bytes and Roboflow augmentation families.
    for split in ("train","valid","test"):
        output_images=destination/("val" if split=="valid" else split)/"images"
        output_labels=output_images.parent/"labels"
        output_images.mkdir(parents=True,exist_ok=True)
        output_labels.mkdir(parents=True,exist_ok=True)
        counted=0
        report["image_hashes_by_split"][split]=[]
        for image,item,annotations,family in all_records[split]:
            binary=image.read_bytes()
            digest=hashlib.sha256(binary).hexdigest()
            if digest in seen_hashes:
                raise ValueError("Imagen duplicada entre particiones")
            # Abort rather than silently leak close variants of a single original.
            if family in family_splits and family_splits[family]!=split:
                raise ValueError("Familia de aumento repartida entre train/val/test: "+family)
            seen_hashes.add(digest)
            family_splits[family]=split
            width,height=int(item["width"]),int(item["height"])
            if width<10 or height<10:raise ValueError("Dimensiones de fotografía inválidas")
            labels=[]
            for a in annotations:
                x,y,w,h=map(float,a["bbox"])
                if min(w,h)<=0 or x<0 or y<0 or x+w>width+2 or y+h>height+2:
                    raise ValueError("Caja de anotación inválida")
                cx=(x+w/2)/width;cy=(y+h/2)/height
                bw=w/width;bh=h/height
                if not all(0<=v<=1 for v in (cx,cy,bw,bh)):raise ValueError("Coordenada fuera de imagen")
                labels.append("0 "+" ".join(f"{v:.8f}" for v in (cx,cy,bw,bh)))
            suffix=image.suffix.lower()
            stem=split+"_"+digest[:18]
            (output_images/(stem+suffix)).write_bytes(binary)
            (output_labels/(stem+".txt")).write_text("\n".join(labels)+"\n",encoding="utf-8")
            report["image_hashes_by_split"][split].append(digest)
            counted+=len(labels)
        report["splits"][split]["boxes"]=counted
    config="path: "+str(destination.resolve())+"\ntrain: train/images\nval: val/images\ntest: test/images\nnames: [pintura_deteriorada]\n"
    (destination/"data.yaml").write_text(config,encoding="utf-8")
    (destination/"provenance.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    return report

def train_public_paint(args):
    from huggingface_hub import snapshot_download
    from ultralytics import YOLO
    import torch
    root=Path(args.cache).resolve()
    root.mkdir(parents=True,exist_ok=True)
    snapshot_download(repo_id=SOURCE,repo_type="dataset",local_dir=str(root),
                      allow_patterns=["README.md","train/*.jpg","train/*.jpeg","train/*.png",
                                      "valid/*.jpg","valid/*.jpeg","valid/*.png",
                                      "test/*.jpg","test/*.jpeg","test/*.png",
                                      "train/*annotations*.json","valid/*annotations*.json",
                                      "test/*annotations*.json"],max_workers=6)
    target=Path(args.output).resolve()
    target.mkdir(parents=True,exist_ok=True)
    data_dir=target/"data"
    audit=convert_coco(root,data_dir,min_per_split=args.min_per_split,max_train=args.max_train)
    print("REAL DATASET AUDIT:",json.dumps({k:v for k,v in audit.items() if k!="image_hashes_by_split"},ensure_ascii=False))
    model=YOLO(args.base)
    if model.task!="detect":raise RuntimeError("La base no es un detector YOLO")
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    model.train(data=str(data_dir/"data.yaml"),epochs=args.epochs,imgsz=args.imgsz,
                batch=args.batch,device=device,workers=0,seed=42,
                project=str(target),name="train",exist_ok=False,
                patience=min(args.epochs,8),plots=False,cache=False,verbose=False)
    best=target/"train"/"weights"/"best.pt"
    if not best.exists():raise RuntimeError("No se generaron pesos entrenados")
    result=YOLO(str(best)).val(data=str(data_dir/"data.yaml"),split="test",
                                     imgsz=args.imgsz,batch=args.batch,
                                     device=device,workers=0,plots=False)
    metrics={"ap50":float(result.box.map50),"recall":float(result.box.mr),
             "precision":float(result.box.mp),"images_train":audit["splits"]["train"]["selected"],
             "images_test":audit["splits"]["test"]["selected"],"epochs":args.epochs,
             "status":"candidato_no_validado_en_peru"}
    gate=metrics["ap50"]>=args.min_ap50 and metrics["recall"]>=args.min_recall
    metrics["technical_gate_passed"]=gate
    (target/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    # Preserve the trained experimental artifact for reproducibility even when the
    # quality gate fails. The separate publication workflow will NEVER release or
    # enable it unless both AP50 and recall pass.
    onnx=Path(YOLO(str(best)).export(format="onnx",imgsz=args.imgsz,opset=17,simplify=False))
    shutil.copy2(onnx,target/"pintura-deteriorada-candidate.onnx")
    metrics["onnx_exported_for_research"]=True
    (target/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print("TEST METRICS:",json.dumps(metrics))
    return metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cache",default="/tmp/paint-public-hf")
    p.add_argument("--output",default="/tmp/paint-yolo-training")
    p.add_argument("--base",default="yolo26n.pt")
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--imgsz",type=int,default=320)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--max-train",type=int,default=260)
    p.add_argument("--min-per-split",type=int,default=5)
    p.add_argument("--min-ap50",type=float,default=0.40)
    p.add_argument("--min-recall",type=float,default=0.50)
    args=p.parse_args()
    if not 1<=args.epochs<=100:raise SystemExit("epochs fuera de rango")
    try:train_public_paint(args)
    except Exception as exc:raise SystemExit("REAL PAINT TRAINING FAILED: "+repr(exc)) from exc

if __name__=="__main__":
    main()
