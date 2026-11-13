"""Visual inspection tests: no external API key required in CI."""
import json
from io import BytesIO

from PIL import Image
from fastapi.testclient import TestClient

from app import app
from vision_service import (
    guided_shots, normalize_vision, evaluate_condition,
    simulate_credit, analyze_photo,
)
import app as server


def make_photo():
    picture = Image.new("RGB", (64, 64), "white")
    output = BytesIO()
    picture.save(output, format="JPEG")
    return output.getvalue()


def test_not_fake_visual_inference_without_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = tmp_path / "image.jpg"
    path.write_bytes(make_photo())
    try:
        analyze_photo(path, "Fachada", "vivienda")
        assert False, "Should refuse to invent model analysis"
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)


def test_normalizer_and_incomplete_score():
    result=normalize_vision({
        "condition": "unknown", "summary":"Test",
        "materials":[{"name":"Ladrillo","location":"Pared"}],
        "issues":[{"category":"Pared sin tarrajear","severity":"moderada",
                   "description":"Ladrillo expuesto","improvement":"Tarrajeo"}],
    })
    assert result["condition"] == "no_determinable"
    assert result["materials"][0]["name"] == "Ladrillo"
    assert result["issues"][0]["severity"] == "moderada"
    empty=evaluate_condition("vivienda", [], [], [])
    assert empty["condition_score"] is None
    assert empty["coverage"] == 0
    assert simulate_credit(empty, 100000, 20000)["suggested_amount"] is None


def test_visual_guided_end_to_end(monkeypatch):
    # Tests the API contract and review decisions, not actual model quality.
    client=TestClient(app)
    result=client.post("/api/login",json={
        "username":"administrador","password":"strong-test-password"})
    assert result.status_code==200
    case=client.post("/api/cases",json={
        "asset_type":"vivienda","title":"Inspección de vivienda"}).json()
    cid=case["id"]
    guide=client.get("/api/inspection-guide/vivienda")
    assert guide.status_code==200
    shots=guide.json()["shots"]
    assert shots==["Fachada","Paredes","Techo","Interior"]
    ids=[]
    for shot in shots:
        uploaded=client.post("/api/cases/"+cid+"/evidence",
            data={"shot":shot},
            files={"file":("casa.jpg",make_photo(),"image/jpeg")})
        assert uploaded.status_code==201, uploaded.text
        ids.append(uploaded.json()["id"])
    case_response=client.get("/api/cases/"+cid).json()
    assert case_response["visual_condition"]["coverage"] == 100
    assert case_response["visual_condition"]["condition_score"] is None

    def dummy_vision(image_path, shot, asset_type):
        assert asset_type=="vivienda"
        return normalize_vision({
            "condition":"requiere_mejoras" if shot=="Paredes" else "aparentemente_bueno",
            "summary":"Observaciones preliminares",
            "materials":[{"name":"Ladrillo visible","location":"Pared"}],
            "issues":[{"category":"Pared sin tarrajear","severity":"moderada",
                       "description":"Acabado incompleto",
                       "improvement":"Solicitar presupuesto de tarrajeo"}] if shot=="Paredes" else [],
            "limitations":["No se puede verificar la estructura internamente"],
        })
    monkeypatch.setattr(server,"analyze_photo",dummy_vision)
    for eid in ids:
        r=client.post("/api/cases/"+cid+"/vision/"+eid)
        assert r.status_code==200, r.text
        assert r.json()["state"]=="pendiente"
    before=client.get("/api/cases/"+cid).json()["visual_condition"]
    assert before["condition_score"] is None
    for assessment in client.get("/api/cases/"+cid).json()["visual_assessments"]:
        r=client.put("/api/visual-assessments/"+assessment["id"]+"/review",
                     json={"state":"confirmado"})
        assert r.status_code==200
    after=client.get("/api/cases/"+cid).json()["visual_condition"]
    assert after["score_status"]=="referencial_revisado"
    assert after["condition_score"]==88
    pending=client.post("/api/cases/"+cid+"/credit-simulation",
                        json={"documented_budget":20000,"budget_source":"Cotización albañil"})
    assert pending.json()["status"]=="pendiente"
    assert client.put("/api/cases/"+cid+"/base",
      json={"base_value":100000,"source":"Comparables documentados"}).status_code==200
    credit=client.post("/api/cases/"+cid+"/credit-simulation",
        json={"documented_budget":20000,"budget_source":"Cotización albañil"})
    assert credit.status_code==200, credit.text
    assert credit.json()["suggested_amount"]==14000
    assert credit.json()["status"]=="simulacion_no_vinculante"
    assert client.get("/api/cases/"+cid+"/report.json").json()["visual_condition"]["condition_score"]==88


def test_one_photo_manual_review_produces_provisional_score():
    client=TestClient(app)
    assert client.post("/api/login",json={"username":"administrador",
                  "password":"strong-test-password"}).status_code==200
    case=client.post("/api/cases",json={"asset_type":"vivienda",
                 "title":"Evaluación con una foto"}).json()
    cid=case["id"]
    upload=client.post("/api/cases/"+cid+"/evidence",data={"shot":"Fachada"},
        files={"file":("fachada.jpg",make_photo(),"image/jpeg")})
    assert upload.status_code==201
    eid=upload.json()["id"]
    r=client.post("/api/cases/"+cid+"/manual-inspection/"+eid,json={
        "condition":"requiere_mejoras","category":"Fisura visible","severity":"moderada",
        "description":"Se observa una grieta sobre el revestimiento",
        "improvement":"Solicitar inspección profesional","material":"Pared revocada",
        "notes":"Observación aportada por el revisor"})
    assert r.status_code==200,r.text
    assert r.json()["result"]["method"]=="manual_review"
    score=client.get("/api/cases/"+cid).json()["visual_condition"]
    assert score["captured"]==1
    assert score["condition_score"]==88
    assert score["score_status"]=="provisional_parcial"
    assert score["credit_ready"] is False
    r=client.post("/api/cases/"+cid+"/credit-simulation",json={
        "documented_budget":1000,"budget_source":"Presupuesto documentado"})
    assert r.json()["suggested_amount"] is None


def test_repeated_manual_review_uses_most_recent():
    client=TestClient(app)
    client.post("/api/login",json={"username":"administrador",
                "password":"strong-test-password"})
    case=client.post("/api/cases",json={"asset_type":"vivienda",
                "title":"Revisión modificada"}).json()
    cid=case["id"]
    eid=client.post("/api/cases/"+cid+"/evidence",data={"shot":"Fachada"},
        files={"file":("fachada.jpg",make_photo(),"image/jpeg")}).json()["id"]
    url="/api/cases/"+cid+"/manual-inspection/"+eid
    assert client.post(url,json={"condition":"requiere_mejoras",
      "category":"Fisura visible","severity":"severa"}).status_code==200
    assert client.post(url,json={"condition":"aparentemente_bueno"}).status_code==200
    result=client.get("/api/cases/"+cid).json()["visual_condition"]
    assert result["condition_score"]==100


def test_yolo_onnx_integration_with_backend(monkeypatch):
    # API route test; real neural inference is separately verified in ONNX CPU Actions.
    from PIL import Image
    client=TestClient(app)
    assert client.post("/api/login",json={
        "username":"administrador","password":"strong-test-password"}).status_code==200
    case=client.post("/api/cases",json={
        "asset_type":"vivienda","title":"Detección automática"}).json()
    cid=case["id"]
    photo=client.post("/api/cases/"+cid+"/evidence",data={"shot":"Fachada"},
                      files={"file":("casa.jpg",make_photo(),"image/jpeg")}).json()
    eid=photo["id"]
    monkeypatch.setenv("YOLO_CPU_ENABLED","1")
    def mock_model(path):
        assert path.endswith(".jpg")
        return [{
            "category":"fisura_visible","confidence":0.94,
            "severity":"leve","origin":"yolo_onnx_fisuras",
            "description":"Posible fisura visible detectada",
            "improvement":"Validar presencia y alcance en visita",
            "box":[8,8,25,50]
        }], Image.new("RGB",(64,64))
    monkeypatch.setattr(server,"predict_cracks",mock_model)
    result=client.post("/api/cases/"+cid+"/automatic-inspection/"+eid)
    assert result.status_code==200, result.text
    body=result.json()
    assert body["state"]=="pendiente"
    assert body["result"]["method"]=="yolo_cpu_onnx"
    assert body["result"]["condition"]=="requiere_mejoras"
    assert len(body["result"]["issues"])==1
    overlay=client.get("/api/evidence/"+eid+"/yolo-overlay")
    assert overlay.status_code==200
    assert overlay.headers["content-type"].startswith("image/jpeg")
    assert client.get("/api/cases/"+cid).json()["visual_condition"]["condition_score"] is None
    assert client.put("/api/visual-assessments/"+body["id"]+"/review",
                      json={"state":"confirmado"}).status_code==200
    condition=client.get("/api/cases/"+cid).json()["visual_condition"]
    assert condition["score_status"]=="provisional_parcial"
    assert condition["condition_score"]==95
    assert condition["credit_ready"] is False


def test_yolo_no_fissure_does_not_mean_safe(monkeypatch):
    from PIL import Image
    client=TestClient(app)
    client.post("/api/login",json={"username":"administrador","password":"strong-test-password"})
    case=client.post("/api/cases",json={"asset_type":"vivienda","title":"Fachada sin fisuras detectadas"}).json()
    eid=client.post("/api/cases/"+case["id"]+"/evidence",data={"shot":"Fachada"},
       files={"file":("casa.jpg",make_photo(),"image/jpeg")}).json()["id"]
    monkeypatch.setenv("YOLO_CPU_ENABLED","1")
    monkeypatch.setattr(server,"predict_cracks",lambda path:([],Image.new("RGB",(64,64))))
    response=client.post("/api/cases/"+case["id"]+"/automatic-inspection/"+eid)
    assert response.status_code==200
    assert response.json()["result"]["condition"]=="no_determinable"
    assert "NO acredita" in response.json()["result"]["summary"]
