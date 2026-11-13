"""YOLO inference adapter. No model is downloaded or executed without explicit configuration."""
import os
from pathlib import Path


def get_model_configuration(asset_type):
    specialized = os.getenv("YOLO_VIVIENDA_WEIGHTS", "").strip()
    generic = os.getenv("YOLO_WEIGHTS", "").strip()
    if asset_type == "vivienda" and specialized:
        return {"path": specialized, "profile": "vivienda_fisuras", "specialized": True}
    if generic:
        return {"path": generic, "profile": "generico", "specialized": False}
    return None


def parse_results(results, specialized=False):
    """Interpret validated inference output; boxes and masks remain observations."""
    findings = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        masks = getattr(result, "masks", None)
        polygons = getattr(masks, "xyn", None) if masks is not None else None
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            names = result.names
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            has_mask = bool(polygons is not None and idx < len(polygons) and len(polygons[idx]) >= 3)
            if specialized:
                category = "fisura_visible" if str(label).lower() in ("crack", "cracks", "fisura", "fissure") else "fisura_visible"
                description = ("Segmentación visual de posible fisura" if has_mask else "Detección visual de posible fisura")
                origin = "yolo_fisuras"
            else:
                category = "objeto_" + str(label)[:90]
                description = "Objeto detectado por modelo genérico; no representa automáticamente un daño"
                origin = "yolo_generico"
            findings.append({
                "category": category,
                "confidence": round(confidence, 5),
                "origin": origin,
                "description": description + "; requiere revisión profesional.",
                "has_mask": has_mask,
            })
    return findings


def predict_findings(image_path, config):
    weights = Path(config["path"]).resolve()
    if not weights.is_file():
        raise FileNotFoundError("No se encuentran los pesos YOLO configurados")
    if weights.suffix.lower() != ".pt":
        raise ValueError("Se esperan pesos YOLO en formato .pt")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Instala requirements-yolo.txt para habilitar inferencia") from exc
    model = YOLO(str(weights))  # Cargar solo pesos de procedencia verificada.
    if config["specialized"] and getattr(model, "task", "") != "segment":
        raise ValueError("El modelo de fisuras debe ser de segmentación, no de detección genérica")
    results = model.predict(str(image_path), conf=0.35, imgsz=640, device="cpu", verbose=False)
    return parse_results(results, config["specialized"])
