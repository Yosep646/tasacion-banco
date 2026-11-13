"""Verify five-defect taxonomy, ONNX postprocessing and the real FastAPI contract."""
import io
import os
os.environ.setdefault('TASACION_ADMIN_PASSWORD','strong-test-password')
os.environ.setdefault('SESSION_SECRET','test-only-session-secret-of-at-least-32-chars')
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
import app
from yolo_defects import unpack_predictions
from inspection_taxonomy import PRETRAINED_5, PENDING_TRAINING, public_capabilities


def test_taxonomy_has_real_model_classes_and_pending_classes():
    assert set(PRETRAINED_5)=={"crack","mold","peeling_paint","stairstep_crack","water_seepage"}
    assert "piso_roto" in PENDING_TRAINING
    assert "hueco_sin_puerta" in PENDING_TRAINING
    assert "hueco_sin_puerta" not in public_capabilities(True)["automatic"]
    assert public_capabilities(False)["automatic"]==["fisura_visible"]


def test_yolo26_raw_output_predictions():
    arr=np.zeros((1,9,8400),dtype=np.float32)
    arr[0,:4,2]=[300,200,160,90]
    arr[0,5,2]=0.87  # mold
    result=unpack_predictions(arr)
    assert len(result)==8400
    assert result[2][2]==1
    assert result[2][3]=="xywh"
    assert result[2][1]==0.87 or abs(result[2][1]-0.87)<1e-6


def test_yolo_legacy_end_to_end_shape():
    arr=np.array([[[20,30,200,250,0.91,3]]],dtype=np.float32)
    out=unpack_predictions(arr)
    assert out[0][2]==3
    assert out[0][3]=="xyxy"


def test_multiclass_auto_analysis_with_mocked_inference(monkeypatch):
    monkeypatch.setenv("YOLO_CPU_ENABLED","1")
    monkeypatch.setenv("YOLO_DEFECT5_ENABLED","1")
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    client=TestClient(app.app)
    assert client.post("/api/login",json={"username":"administrador","password":"strong-test-password"}).status_code==200
    cap=client.get("/api/model-capabilities")
    assert cap.status_code==200
    assert "moho_visible" in cap.json()["automatic"]
    image=io.BytesIO()
    Image.new("RGB",(100,100),"white").save(image,format="JPEG")

    def fake(path):
        return [
            {"category":"humedad_visible","confidence":0.67,"description":"Posible mancha de filtración",
             "improvement":"Reparar filtración","box":[2,3,50,65]},
            {"category":"pintura_desprendida","confidence":0.64,"description":"Pintura desprendida",
             "improvement":"Repintar tras preparar","box":[6,8,60,70]},
        ],Image.new("RGB",(100,100),"white")

    monkeypatch.setattr(app,"detect_defects",fake)
    result=client.post("/api/quick-inspection",files={
        "exterior":("foto.jpg",image.getvalue(),"image/jpeg")})
    assert result.status_code==201,result.text
    body=result.json()
    assert body["images_analyzed"]==1
    assert {d["category"] for d in body["issues"]}=={"humedad_visible","pintura_desprendida"}
    assert body["quote"]["total_pen"] is None
    assert body["property_market_value"] is None
    assert body["human_review_required"]
    assert body["image_summaries"][0]["annotated_url"].endswith("/yolo-overlay")
    assert client.get("/api/cases/"+body["case_id"]+"/quick-report.pdf").status_code==200


def test_model_failure_falls_back_to_previous_cracks(monkeypatch):
    monkeypatch.setenv("YOLO_CPU_ENABLED","1")
    monkeypatch.setenv("YOLO_DEFECT5_ENABLED","1")
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    client=TestClient(app.app)
    client.post("/api/login",json={"username":"administrador","password":"strong-test-password"})
    def failed(path):raise RuntimeError("missing multicolor")
    monkeypatch.setattr(app,"detect_defects",failed)
    monkeypatch.setattr(app,"predict_cracks",lambda p:(
        [{"category":"fisura_visible","confidence":0.84,"description":"Posible fisura",
          "improvement":"Inspección presencial","box":[3,3,40,40]}],Image.new("RGB",(50,50))))
    buffer=io.BytesIO()
    Image.new("RGB",(50,50)).save(buffer,format="JPEG")
    response=client.post("/api/quick-inspection",files={"exterior":("x.jpg",buffer.getvalue(),"image/jpeg")})
    assert response.status_code==201,response.text
    report=response.json()
    assert len(report["issues"])==1
    assert report["issues"][0]["category"]=="fisura_visible"
    assert report["analysis_errors"]
    assert "fisuras" in " ".join(report["limitations"]).lower() or "fisuras" in " ".join(report["image_summaries"]).lower()
