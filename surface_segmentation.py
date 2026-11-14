"""CPU-only semantic segmentation of walls, floors and ceilings.

SegFormer-B0 ADE20K (150 labels) in pinned, quantized ONNX format.
This is scene segmentation, NOT material recognition or a physical area survey.
No OpenAI token, PyTorch or user-uploaded images leave this service.
"""
import hashlib
import os
import threading
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_NAME = "segformer-b0-ade20k-int8.onnx"
MODEL_URL = (
    "https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/"
    "resolve/main/onnx/model_quantized.onnx"
)
MODEL_SHA256 = "9a98d6daf3d926869ab8cc4c2ed7374a2bc23b889bb7ca3b0915d15e3c4756bb"
MODEL_SOURCE = "https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512"
INPUT_SIZE = (512, 512)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Official ADE20K 150-class mapping from the upstream SegFormer config.
SURFACES = {
    "pared": {"class_id": 0, "label": "Pared", "rgb": (42, 126, 225)},
    "piso": {"class_id": 3, "label": "Piso", "rgb": (25, 159, 117)},
    "techo": {"class_id": 5, "label": "Techo / cielo raso", "rgb": (242, 154, 46)},
}
_session = None
_lock = threading.Lock()


def model_path():
    folder = Path(os.getenv("TASACION_MODEL_DIR", "/tmp/tasacion-models"))
    return folder / MODEL_NAME


def download_model():
    path = model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA256:
        return path
    part = path.with_name(path.name + ".partial")
    try:
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(MODEL_URL, timeout=80) as response, part.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 12 * 1024 * 1024:
                    raise RuntimeError("Los pesos de segmentación exceden el límite previsto")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != MODEL_SHA256:
            raise RuntimeError("La verificación SHA-256 del modelo de superficies falló")
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
    return path


def get_session():
    global _session
    with _lock:
        if _session is None:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _session = ort.InferenceSession(
                str(download_model()), sess_options=options, providers=["CPUExecutionProvider"]
            )
    return _session


def preprocess(image):
    """ImageNet normalization and 512x512 bilinear scaling from model processor."""
    resized = image.resize(INPUT_SIZE, Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(image_array.transpose(2, 0, 1)[None])


def render_surfaces(image, logits):
    """Return semantic pixel masks, transparent PNG and overlay; no metre values."""
    values = np.asarray(logits)
    if values.ndim == 4 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 3 or values.shape[0] != 150 or not np.isfinite(values).all():
        raise RuntimeError("Salida de segmentación ONNX inesperada")
    categories = np.argmax(values, axis=0).astype(np.uint8)
    resampled = Image.fromarray(categories, "L").resize(image.size, Image.Resampling.NEAREST)
    labels = np.asarray(resampled)
    height, width = labels.shape
    pixel_total = height * width
    overlay = image.copy()
    semantic_mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
    classes = []
    for key, info in SURFACES.items():
        pixels = labels == info["class_id"]
        count = int(np.count_nonzero(pixels))
        percentage = round(100 * count / pixel_total, 2)
        mask = Image.fromarray(np.where(pixels, 255, 0).astype(np.uint8), "L")
        color = info["rgb"]
        blended = Image.fromarray(np.where(pixels, 85, 0).astype(np.uint8), "L")
        overlay.paste(Image.new("RGB", image.size, color), (0, 0), blended)
        semantic_mask.paste(
            Image.new("RGBA", image.size, (*color, 190)), (0, 0), mask
        )
        classes.append({
            "key": key, "label": info["label"], "pixels": count,
            "percentage_image": percentage,
            "present": percentage >= 0.5,
            "color": "#" + "".join(f"{v:02x}" for v in color),
        })
    return ({
        "status": "preliminar_por_verificar",
        "method": "segformer_b0_ade20k_onnx",
        "model": "SegFormer-B0 (ADE20K, ONNX INT8)",
        "model_source": MODEL_SOURCE,
        "image_width_px": width,
        "image_height_px": height,
        "classes": classes,
        "coverage_unit": "porcentaje_de_pixeles_de_la_fotografia",
        "limitations": [
            "Segmentación semántica orientativa; verificar visualmente los límites.",
            "Una foto no permite calcular m² ni deducir dimensiones reales.",
            "Una pared o techo segmentado no demuestra sus materiales ni su estado estructural.",
            "Las superficies ocultas o fuera de cuadro no pueden inferirse.",
        ],
    }, overlay, semantic_mask)


def segment_surfaces(image_path):
    """Predict walls, floor and ceiling. Return (metadata, JPEG overlay, RGBA PNG mask)."""
    with Image.open(image_path) as file:
        image = file.convert("RGB")
    model = get_session()
    input_tensor = preprocess(image)
    output = model.run(None, {model.get_inputs()[0].name: input_tensor})
    if not output:
        raise RuntimeError("El modelo de superficies no devolvió resultados")
    return render_surfaces(image, output[0])
