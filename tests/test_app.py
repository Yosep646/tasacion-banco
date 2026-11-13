import importlib
import os

os.environ["APP_ENV"] = "development"
os.environ["TASACION_ADMIN_PASSWORD"] = "strong-test-password"
os.environ["SESSION_SECRET"] = "local-testing-secret-do-not-use-in-production"
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_auth_required():
    assert client.get("/api/cases").status_code == 401

def test_create_and_review():
    assert client.post("/api/login",json={"username":"administrador","password":"strong-test-password"}).status_code == 200
    r=client.post("/api/cases",json={"asset_type":"vivienda","title":"Vivienda muestra"})
    assert r.status_code == 201, r.text
    case_id=r.json()["id"]
    r=client.put("/api/cases/"+case_id+"/base",json={"base_value":100000,"source":"Comparables documentados"})
    assert r.status_code == 200
    r=client.post("/api/cases/"+case_id+"/findings",json={"category":"fisura","severity":"moderada","description":"Observación visual"})
    assert r.status_code == 201
    fid=r.json()["id"]
    r=client.put("/api/findings/"+fid+"/review",json={"state":"confirmado","adjustment":5000,"justification":"Presupuesto de reparación"})
    assert r.status_code == 200
    r=client.get("/api/cases/"+case_id)
    assert r.json()["estimated_value"] == 95000
    assert client.get("/api/cases/"+case_id+"/report.pdf").content.startswith(b"%PDF")
    assert client.get("/api/cases/"+case_id+"/report.json").status_code == 200
    assert client.get("/api/cases/"+case_id+"/audit").status_code == 200
