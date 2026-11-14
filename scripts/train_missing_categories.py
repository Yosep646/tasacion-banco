"""Supervised YOLO training for NINE not-yet-supported housing classes.

Requires real, permissioned, human-reviewed labels. Never publishes an
unvalidated model or fabricates accuracy metrics.
"""
from pathlib import Path
import argparse
import json

from prepare_housing_dataset import CLASSES


def validate_dataset_yaml(path):
    import yaml
    if not path.is_file():
        raise ValueError("No existe data.yaml con imágenes y anotaciones reales")
    data=yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data,dict) or not {"train","val","test","names"}.issubset(data):
        raise ValueError("Faltan train/val/test o nombres de clases")
    names=data["names"]
    names=[names[i] if i in names else names[str(i)] for i in range(len(names))] if isinstance(names,dict) else list(names)
    if tuple(names)!=CLASSES:
        raise ValueError("Orden de clases incompatible con el detector")
    root=Path(data.get("path",path.parent))
    if not root.is_absolute():
        root=(path.parent/root).resolve()
    for split in ("train","val","test"):
        images=(root/data[split]).resolve()
        if not images.is_dir() or not any(images.iterdir()):
            raise ValueError("Partición inexistente o vacía: "+split)
    manifest=path.parent/"dataset_manifest.json"
    if not manifest.exists():
        raise ValueError("Falta manifiesto de procedencia y licencias")
    info=json.loads(manifest.read_text(encoding="utf-8"))
    if info.get("status")!="dataset_prepared_not_trained":
        raise ValueError("Dataset no superó controles de preparación")
    return data


def audit_metrics(metrics,min_ap50=0.40,min_recall=0.50):
    """Require independent test AP50 and recall for EVERY class."""
    ap=list(map(float,metrics.box.ap50))
    recall=list(map(float,metrics.box.r))
    indices=list(map(int,metrics.box.ap_class_index))
    items={c:{"ap50":ap[i],"recall":recall[i]} for i,c in enumerate(indices)}
    if set(items)!=set(range(len(CLASSES))):
        raise ValueError("Una o más clases no tienen ejemplos en la evaluación")
    passed=all(v["ap50"]>=min_ap50 and v["recall"]>=min_recall for v in items.values())
    return {"passed_per_class":passed,"minimum_ap50":min_ap50,
            "minimum_recall":min_recall,
            "class_metrics":{CLASSES[i]:items[i] for i in range(len(CLASSES))},
            "mAP50":float(metrics.box.map50),
            "status":"passed_technical_gate_not_field_validated" if passed else "needs_retraining"}


def train(args):
    yaml=Path(args.dataset).resolve()
    validate_dataset_yaml(yaml)
    from ultralytics import YOLO
    import torch
    device=0 if torch.cuda.is_available() else "cpu"
    base=YOLO(args.weights)
    if base.task!="detect":
        raise ValueError("El modelo base no es detector de cajas")
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    base.train(data=str(yaml),epochs=args.epochs,imgsz=args.imgsz,
               project=str(output),name="housing-nine",exist_ok=False,
               device=device,workers=2 if device!="cpu" else 0,
               patience=max(5,args.epochs//5),seed=42,plots=True,cache=False)
    best=output/"housing-nine"/"weights"/"best.pt"
    if not best.is_file():
        raise RuntimeError("No se generaron pesos best.pt")
    trained=YOLO(str(best))
    test=trained.val(data=str(yaml),split="test",imgsz=args.imgsz,
                     device=device,project=str(output),name="holdout-test",plots=True)
    report=audit_metrics(test,args.min_ap50,args.min_recall)
    report.update({"source_dataset":str(yaml),"epochs":args.epochs,
                   "weights":str(best),"exported":False,
                   "field_validation":"pending","license_audit":"pending"})
    out=output/"evaluation_report.json"
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if not report["passed_per_class"]:
        raise RuntimeError("Falla la métrica por clase; no publicar ni activar")
    candidate=Path(trained.export(format="onnx",imgsz=args.imgsz,opset=17,simplify=False))
    (output/"candidate_model.onnx").write_bytes(candidate.read_bytes())
    report["exported"]=True
    report["status"]="candidate_pending_field_and_license_review"
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--epochs",type=int,default=80)
    parser.add_argument("--imgsz",type=int,default=640)
    parser.add_argument("--weights",default="yolo26n.pt")
    parser.add_argument("--output",default="runs/housing-n9")
    parser.add_argument("--min-ap50",type=float,default=0.40)
    parser.add_argument("--min-recall",type=float,default=0.50)
    args=parser.parse_args()
    if not 1<=args.epochs<=300 or not 320<=args.imgsz<=1280:
        raise SystemExit("Parámetros fuera de rango")
    try:train(args)
    except (OSError,RuntimeError,ValueError) as exc:
        raise SystemExit("TRAINING BLOCKED: "+str(exc)) from exc

if __name__=="__main__":main()
