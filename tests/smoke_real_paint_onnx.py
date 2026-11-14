"""Smoke-test a genuinely published experimental paint ONNX with the real runtime.

This test uses a generated fixture ONLY to check software interoperability.
It is not accuracy, field, or credit validation.
"""
import io
import os
import tempfile
from pathlib import Path
import sys
from PIL import Image, ImageDraw

os.environ["APP_ENV"]="development"
os.environ["TASACION_ADMIN_PASSWORD"]="test-password-not-for-production"
os.environ["SESSION_SECRET"]="smoke-validation-secret-32-chars-plus-0123456789"
os.environ["YOLO_PAINT_ENABLED"]="1"
os.environ["YOLO_PAINT_SHA256"]="c20c6a01fac4fc32bfade1a35711c8b57cef1c9f648afc9ad0fc4e273fdd1f5d"
os.environ["YOLO_CPU_ENABLED"]="1"
os.environ.pop("YOLO_DEFECT5_ENABLED",None)
os.environ.pop("OPENAI_API_KEY",None)
temp=tempfile.TemporaryDirectory()
os.environ["TASACION_DATA"]=temp.name
os.environ["TASACION_MODEL_DIR"]=str(Path(temp.name)/"models")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import app
from yolo_paint import detect_paint, get_model
from fastapi.testclient import TestClient

image=Image.new("RGB",(416,416),"#dbccaf")
d=ImageDraw.Draw(image)
d.polygon([(80,50),(170,46),(159,84),(115,97)],fill="#beb6a6")
d.line([(65,165),(130,155),(122,205),(197,230)],fill="#796d5d",width=4)
path=Path(temp.name)/"synthetic-paint-test.jpg"
image.save(path)
file=get_model()
print("Downloaded SHA256-pinned experimental model:",file.stat().st_size,"bytes")
suggestions,overlay=detect_paint(str(path))
assert isinstance(suggestions,list)
assert overlay.size==image.size
assert all(item["category"]=="pintura_deteriorada" for item in suggestions)
assert len(suggestions)<=15
print("Real CPU inference:",len(suggestions),"candidate results on software fixture")
# Mock only the previously verified primary crack model to isolate the new
# experimental paint model's real inference through FastAPI.
app.predict_cracks=lambda p:([],Image.open(p).convert("RGB"))
with TestClient(app.app) as client:
    login=client.post("/api/login",json={"username":"administrador","password":"test-password-not-for-production"})
    assert login.status_code==200,login.text
    readiness=client.get("/api/model-readiness")
    assert readiness.status_code==200
    assert readiness.json()["yolo_paint"]["configured"] is True
    buf=io.BytesIO()
    image.save(buf,format="JPEG")
    res=client.post("/api/quick-inspection",files={"exterior":("pared.jpg",buf.getvalue(),"image/jpeg")})
    assert res.status_code==201,res.text
    report=res.json()
    assert not report["analysis_errors"],report["analysis_errors"]
    assert report["human_review_required"]
    assert report["quote"]["total_pen"] is None
    assert report["property_market_value"] is None
    assert client.get("/api/cases/"+report["case_id"]+"/quick-report.pdf").status_code==200
print("PASS: published ONNX -> actual runtime -> FastAPI -> PDF. No field accuracy claim.")
temp.cleanup()
