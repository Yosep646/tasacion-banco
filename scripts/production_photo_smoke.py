"""Isolated production-runtime end-to-end test with two CC0 photographs.

Runs inside Render only when PRODUCTION_PHOTO_SMOKE_ENABLED=1 on startup.
Uses a disposable SQLite DB and a random test password: never touches
real client files, production cases, or OpenAI.
"""
import io
import json
import os
import secrets
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image

SAMPLES = (
    ("Exterior", "https://upload.wikimedia.org/wikipedia/commons/6/62/House_facade_2023.jpg",
     "https://commons.wikimedia.org/wiki/File:House_facade_2023.jpg"),
    ("Interior", "https://upload.wikimedia.org/wikipedia/commons/7/73/Living_Room_Interior_Designer_-_Picker_Online.jpg",
     "https://commons.wikimedia.org/wiki/File:Living_Room_Interior_Designer_-_Picker_Online.jpg"),
)


def download_real_photo(url):
    request = urllib.request.Request(url, headers={
        "User-Agent": "Tasacion-Banco-Image-QA/1.0 (CC0 Wikimedia Commons; validation only)",
        "Accept": "image/jpeg,image/*",
    })
    with urllib.request.urlopen(request, timeout=25) as response:
        image_bytes = response.read(5 * 1024 * 1024 + 1)
    if len(image_bytes) > 5 * 1024 * 1024:
        raise RuntimeError("Demo image exceeds allowed size")
    with Image.open(io.BytesIO(image_bytes)) as picture:
        picture.verify()
    with Image.open(io.BytesIO(image_bytes)) as picture:
        image = picture.convert("RGB")
        image.thumbnail((900, 900))
        output = io.BytesIO()
        image.save(output, "JPEG", quality=82)
        return output.getvalue()


def main():
    from fastapi.testclient import TestClient

    with tempfile.TemporaryDirectory(prefix="tasacion-prod-smoke-") as temp:
        # Configure isolation before importing app (init_db runs at import time).
        os.environ.update({
            "TASACION_DATA": temp,
            "TASACION_DB": str(Path(temp) / "cases.sqlite3"),
            "SESSION_SECRET": secrets.token_urlsafe(48),
            "TASACION_ADMIN_PASSWORD": secrets.token_urlsafe(32),
            "APP_ENV": "development",
            "SURFACE_SEGMENTATION_ENABLED": "1",
            "YOLO_CPU_ENABLED": "0",
            "YOLO_DEFECT5_ENABLED": "0",
            "YOLO_PAINT_ENABLED": "0",
            "PRODUCTION_PHOTO_SMOKE_ENABLED": "0",
        })
        # Never use the user's exposed API credential, call OpenAI, or incur API charges.
        os.environ.pop("OPENAI_API_KEY", None)
        for secret_key in ("TASACION_REVISOR_PASSWORD", "TASACION_OPERADOR_PASSWORD"):
            os.environ.pop(secret_key, None)

        import app as server

        photos = [(shot, download_real_photo(url), reference)
                  for shot, url, reference in SAMPLES]
        client = TestClient(server.app)
        assert client.get("/api/cases").status_code == 401
        login = client.post("/api/login", json={
            "username": "administrador",
            "password": os.environ["TASACION_ADMIN_PASSWORD"],
        })
        assert login.status_code == 200, f"Login failed: {login.status_code}"
        files = [
            ("exterior", ("facade.jpg", photos[0][1], "image/jpeg")),
            ("interior", ("interior.jpg", photos[1][1], "image/jpeg")),
        ]
        response = client.post("/api/quick-inspection", files=files)
        assert response.status_code == 201, (
            f"Quick inspection HTTP {response.status_code}: {response.text[:160]}")
        report = response.json()
        assert report["images_analyzed"] == 2
        assert report["property_market_value"] is None
        assert report["human_review_required"] is True
        assert len(report["image_summaries"]) == 2

        evaluations = []
        for summary in report["image_summaries"]:
            surfaces = summary.get("surfaces")
            assert surfaces and len(surfaces["classes"]) == 3
            assert surfaces["method"] == "segformer_b0_ade20k_onnx"
            evidence_id = summary["evidence_id"]
            overlay = client.get(f"/api/evidence/{evidence_id}/surface-overlay")
            mask = client.get(f"/api/evidence/{evidence_id}/surface-mask")
            assert overlay.status_code == 200 and overlay.headers["content-type"] == "image/jpeg"
            assert mask.status_code == 200 and mask.headers["content-type"] == "image/png"
            evaluations.append({
                "view": summary["shot"],
                "class_names": [v["key"] for v in surfaces["classes"]],
                "percentages_image": {
                    v["key"]: v["percentage_image"] for v in surfaces["classes"]},
            })

        cid = report["case_id"]
        saved = client.get(f"/api/cases/{cid}/quick-report")
        assert saved.status_code == 200 and saved.json()["images_analyzed"] == 2
        case = client.get(f"/api/cases/{cid}")
        assert case.status_code == 200
        assert len(case.json()["evidence"]) == 2
        pdf = client.get(f"/api/cases/{cid}/quick-report.pdf")
        assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
        print("PRODUCTION_SMOKE_RESULT=" + json.dumps({
            "result": "pass",
            "photographs_real": 2,
            "source": "Wikimedia Commons CC0",
            "model": "SegFormer-B0 ADE20K ONNX",
            "api_upload": True,
            "api_persisted": True,
            "pdf_generated": True,
            "overlays_downloadable": True,
            "review_required": True,
            "samples": evaluations,
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
