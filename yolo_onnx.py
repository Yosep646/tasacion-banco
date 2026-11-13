"""Small CPU ONNX runtime for YOLOv8-seg cracks, no torch or ultralytics at runtime."""
import hashlib
import os
import threading
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BASE = "https://github.com/Yosep646/tasacion-banco/releases/download/v0.1-crack-onnx/"
NAME = "vivienda-fisuras.onnx"
_session = None
_lock = threading.Lock()


def model_dir():
    return Path(os.getenv("TASACION_MODEL_DIR", "/tmp/tasacion-models"))


def ensure_model():
    folder = model_dir()
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / NAME
    sha_file = folder / (NAME + ".sha256")
    # SHA256 is downloaded from the versioned release. Verify every download.
    if not sha_file.is_file():
        with urllib.request.urlopen(BASE + NAME + ".sha256", timeout=25) as source:
            expected_line = source.read(160).decode("ascii").strip()
        expected = expected_line.split()[0].lower()
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise RuntimeError("Checksum del modelo inválido")
        sha_file.write_text(expected, encoding="ascii")
    else:
        expected = sha_file.read_text(encoding="ascii").strip()
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == expected:
        return dest
    partial = folder / (NAME + ".partial")
    try:
        digest = hashlib.sha256()
        with urllib.request.urlopen(BASE + NAME, timeout=90) as source, partial.open("wb") as out:
            length = 0
            while chunk := source.read(1024 * 1024):
                length += len(chunk)
                if length > 45 * 1024 * 1024:
                    raise ValueError("Modelo superior a tamaño esperado")
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError("Integridad del modelo ONNX no válida")
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)
    return dest


def get_session():
    global _session
    with _lock:
        if _session is None:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _session = ort.InferenceSession(str(ensure_model()), sess_options=options,
                                            providers=["CPUExecutionProvider"])
    return _session


def _sigmoid(a):
    return 1.0 / (1.0 + np.exp(-np.clip(a, -40, 40)))


def predict_cracks(image_path, confidence=0.35):
    session = get_session()
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
    w, h = rgb.size
    scale = min(640 / w, 640 / h)
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    dx, dy = (640-rw)//2, (640-rh)//2
    canvas = Image.new("RGB", (640,640), (114,114,114))
    canvas.paste(rgb.resize((rw,rh), Image.Resampling.BILINEAR), (dx,dy))
    arr = np.asarray(canvas, dtype=np.float32)
    input_image = arr.transpose(2,0,1)[None] / 255.0
    outputs = session.run(None, {session.get_inputs()[0].name:input_image})
    preds = outputs[0]
    if preds.ndim != 3:
        raise RuntimeError("Formato de predicción YOLO incompatible")
    preds = preds[0]
    # YOLOv8-seg: [4 xywh, number_of_classes, 32 mask_coefficients, 8400 anchors]
    if preds.shape[0] > preds.shape[1]:
        preds = preds.T
    class_count = preds.shape[0] - 4 - 32
    if class_count < 1 or class_count > 40:
        raise RuntimeError("Modelo no coincide con YOLOv8-seg")
    scores = preds[4:4+class_count].max(axis=0)
    indices = np.where(scores >= confidence)[0]
    indices = indices[np.argsort(scores[indices])[::-1]][:60]
    found = []
    for i in indices:
        mx,my,bw,bh = map(float,preds[:4,i])
        xyxy = [mx-bw/2,my-bh/2,mx+bw/2,my+bh/2]
        if bw <= 0 or bh <= 0:
            continue
        # Non-maximum suppression to prevent repeated observations.
        skip = False
        for prev in found:
            other = prev["raw_box"]
            ix1,iy1,ix2,iy2 = max(xyxy[0],other[0]),max(xyxy[1],other[1]),min(xyxy[2],other[2]),min(xyxy[3],other[3])
            intersection = max(0, ix2-ix1)*max(0, iy2-iy1)
            area1 = max(0, xyxy[2]-xyxy[0])*max(0, xyxy[3]-xyxy[1])
            area2 = max(0, other[2]-other[0])*max(0, other[3]-other[1])
            if intersection/max(area1+area2-intersection,1e-6) > 0.45:
                skip = True
                break
        if skip:
            continue
        orig_box = [
            max(0, min(w, (xyxy[0]-dx)/scale)),
            max(0, min(h, (xyxy[1]-dy)/scale)),
            max(0, min(w, (xyxy[2]-dx)/scale)),
            max(0, min(h, (xyxy[3]-dy)/scale)),
        ]
        if orig_box[2] <= orig_box[0] or orig_box[3] <= orig_box[1]:
            continue
        found.append({"confidence":round(float(scores[i]),4),
                      "category":"fisura_visible",
                      "severity":"leve",
                      "origin":"yolo_onnx_fisuras",
                      "description":"Posible fisura detectada automáticamente; requiere verificación.",
                      "improvement":"Inspección presencial y presupuesto de reparación si corresponde",
                      "raw_box":xyxy, "box":orig_box})
        if len(found) >= 12:
            break
    # Draw supported bounding boxes. These are not structural damage masks.
    annotated = rgb.copy()
    draw = ImageDraw.Draw(annotated)
    for item in found:
        draw.rectangle(tuple(item["box"]), outline=(224,37,44), width=max(2,int(w/300)))
        draw.text((item["box"][0],max(0,item["box"][1]-18)),
                  "Posible fisura "+str(round(item["confidence"]*100))+"%",
                  fill=(224,37,44))
    for item in found:
        item.pop("raw_box", None)
    return found, annotated
