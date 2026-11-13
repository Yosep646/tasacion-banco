"""Training entrypoint for missing housing categories; requires labeled, permissioned data.

Never describe this script as a trained model. It trains only when a real
dataset exists and exports ONNX for CPU inference after evaluation.
"""
from pathlib import Path
import argparse
import json

REQUIRED_CLASSES = [
    "piso_roto","pared_sin_tarrajeo","pared_tarrajeada",
    "ladrillo_expuesto","puerta_visible","hueco_sin_puerta",
    "calamina_visible","techo_concreto_visible","pintura_deteriorada",
]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True,help="data.yaml de Ultralytics con train/val/test")
    ap.add_argument("--epochs",type=int,default=80)
    ap.add_argument("--weights",default="yolo26n.pt")
    ap.add_argument("--output",default="runs/housing-defects")
    args=ap.parse_args()
    import yaml
    dataset=Path(args.dataset).resolve()
    if not dataset.is_file():raise SystemExit("Dataset no encontrado")
    data=yaml.safe_load(dataset.read_text(encoding="utf-8"))
    names=data.get("names",{})
    names=list(names.values()) if isinstance(names,dict) else list(names)
    if sorted(names)!=sorted(REQUIRED_CLASSES):
        raise SystemExit("Clases no coinciden con la taxonomía requerida")
    if "test" not in data or "val" not in data:
        raise SystemExit("Se requieren particiones de validación y prueba independientes")
    from ultralytics import YOLO
    model=YOLO(args.weights)
    model.train(data=str(dataset),epochs=args.epochs,imgsz=640,project=args.output,seed=42,
                patience=18,workers=2)
    val=model.val(data=str(dataset),split="test")
    p50=float(val.box.map50)
    print(json.dumps({"mAP50_test":p50,"classes":names}))
    # Threshold illustrates an acceptance gate; additionally inspect each class's AP,
    # calibration, confusion matrix and false-negative rate on Peruvian houses.
    if p50<0.60:
        raise SystemExit("Modelo no cumple el umbral experimental mAP50 0.60")
    print("Modelo en evaluación: todavía requiere auditoría por clase antes de habilitarlo")
    # Do not auto-export to production or enable Render from an unreviewed training run.

if __name__=="__main__":
    main()
