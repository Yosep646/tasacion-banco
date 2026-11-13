"""Run actual inference against SHA256-verified YOLOv8n-seg crack model."""
import json
import os
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw
from yolo_service import predict_findings


def main():
    model_path = os.environ.get("YOLO_VIVIENDA_WEIGHTS", "weights/vivienda-fisuras.pt")
    assert Path(model_path).is_file(), "Falta modelo especializado"
    with tempfile.TemporaryDirectory() as folder:
        img_path = Path(folder) / "synthetic.jpg"
        img = Image.new("RGB", (640, 640), "#dbd8d1")
        draw = ImageDraw.Draw(img)
        draw.line([(65, 15), (120, 180), (370, 330), (410, 620)], fill="#363430", width=3)
        img.save(img_path)
        results = predict_findings(str(img_path), {
            "path": model_path, "profile": "vivienda_fisuras", "specialized": True
        })
        assert isinstance(results, list)
        assert all(f["origin"] == "yolo_fisuras" for f in results)
        print(json.dumps({"inference": "success", "image": "synthetic_crack",
                          "detections": len(results), "model": "YOLOv8n-seg"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
