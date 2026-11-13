"""End-to-end real YOLO smoke test through the FastAPI service, no mock inference."""
import io
import os
import tempfile
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["APP_ENV"]="development"
os.environ["TASACION_ADMIN_PASSWORD"]="ci-only-test-password-not-for-production"
os.environ["SESSION_SECRET"]="integration-test-secret-at-least-32-characters"
os.environ["YOLO_CPU_ENABLED"]="1"
_temp=tempfile.TemporaryDirectory()
os.environ["TASACION_DATA"]=_temp.name

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from app import app

image=Image.new("RGB",(640,640),"#dbd8d1")
draw=ImageDraw.Draw(image)
draw.line([(65,15),(120,180),(370,330),(410,620)],fill="#363430",width=3)
buffer=io.BytesIO()
image.save(buffer,format="JPEG")

with TestClient(app) as client:
    assert client.get("/health").json()["yolo_cpu_enabled"] is True
    assert client.get("/api/cases").status_code==401
    login=client.post("/api/login",json={"username":"administrador",
                    "password":"ci-only-test-password-not-for-production"})
    assert login.status_code==200, login.text
    case=client.post("/api/cases",json={"asset_type":"vivienda","title":"Ensayo real YOLO"})
    assert case.status_code==201, case.text
    cid=case.json()["id"]
    upload=client.post(f"/api/cases/{cid}/evidence",data={"shot":"Fachada"},
               files={"file":("fachada.jpg",buffer.getvalue(),"image/jpeg")})
    assert upload.status_code==201, upload.text
    eid=upload.json()["id"]
    analysis=client.post(f"/api/cases/{cid}/automatic-inspection/{eid}")
    assert analysis.status_code==200, analysis.text
    result=analysis.json()
    assert result["state"]=="pendiente"
    assert result["result"]["method"]=="yolo_cpu_onnx"
    assert len(result["result"]["issues"])>=1, "Synthetic fissure should be detected"
    assert result["result"]["condition"]=="requiere_mejoras"
    overlay=client.get(f"/api/evidence/{eid}/yolo-overlay")
    assert overlay.status_code==200 and overlay.content.startswith(b"\xff\xd8")
    review=client.put(f"/api/visual-assessments/{result['id']}/review",
                      json={"state":"confirmado"})
    assert review.status_code==200
    report=client.get(f"/api/cases/{cid}/report.json")
    assert report.status_code==200
    assert report.json()["visual_condition"]["score_status"]=="provisional_parcial"
    assert report.json()["visual_condition"]["credit_ready"] is False
    print(f"PASS: real YOLO -> {len(result['result']['issues'])} findings, overlay, review, score & report.")
_temp.cleanup()
