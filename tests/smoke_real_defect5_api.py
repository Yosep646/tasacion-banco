"""Real CPU ONNX five-class inference through the authenticated FastAPI workflow."""
import io
import os
import sys
import tempfile
from pathlib import Path
os.environ["TASACION_ADMIN_PASSWORD"]="isolated-smoke-test-password"
os.environ["SESSION_SECRET"]="isolated-smoke-secret-for-local-tests-only-123"
os.environ["APP_ENV"]="development"
os.environ["YOLO_CPU_ENABLED"]="1"
os.environ["YOLO_DEFECT5_ENABLED"]="1"
folder=tempfile.TemporaryDirectory()
os.environ["TASACION_DATA"]=folder.name
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from PIL import Image,ImageDraw
from app import app

im=Image.new("RGB",(640,640),"#e4ded1")
pen=ImageDraw.Draw(im)
pen.line([(110,70),(174,130),(210,250),(392,420),(430,570)],fill="#33332f",width=5)
buf=io.BytesIO()
im.save(buf,format="JPEG")
with TestClient(app) as client:
    assert client.get("/health").json()["multiclass_yolo_enabled"] is True
    assert client.get("/api/cases").status_code==401
    login=client.post("/api/login",json={"username":"administrador","password":"isolated-smoke-test-password"})
    assert login.status_code==200,login.text
    result=client.post("/api/quick-inspection",files={"exterior":("pared.jpg",buf.getvalue(),"image/jpeg")})
    assert result.status_code==201,result.text
    data=result.json()
    assert data["images_analyzed"]==1
    assert "yolo26_defect5" in data["models"]
    assert data["image_summaries"][0]["annotated_url"]
    assert data["status"]=="preliminar_automatico"
    assert data["property_market_value"] is None
    assert data["human_review_required"]
    assert set(data["capabilities"]["automatic"]) >= {"moho_visible","pintura_desprendida","humedad_visible"}
    assert len(data["issues"])>=1, "Se esperaba una posible fisura en la imagen sintética conocida"
    assert client.get("/api/cases/"+data["case_id"]+"/quick-report.pdf").content.startswith(b"%PDF")
    print("PASS: Real five-class YOLO > FastAPI > Image > "+str(len(data["issues"]))+" findings > Report > PDF")
folder.cleanup()
