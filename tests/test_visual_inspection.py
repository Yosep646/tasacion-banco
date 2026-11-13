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
