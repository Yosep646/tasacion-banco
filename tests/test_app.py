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


def test_upload_and_yolo_disabled():
    from io import BytesIO
    from PIL import Image
    image = Image.new("RGB", (32, 32), "white")
    buf = BytesIO()
    image.save(buf, format="JPEG")
    client.post("/api/login", json={"username": "administrador", "password": "strong-test-password"})
    case = client.post("/api/cases", json={"asset_type":"vivienda", "title":"Prueba evidencia"}).json()
    ident = case["id"]
    up = client.post("/api/cases/"+ident+"/evidence",
                     data={"shot":"Detalles"},
                     files={"file":("foto.jpg",buf.getvalue(),"image/jpeg")})
    assert up.status_code == 201, up.text
    eid=up.json()["id"]
    assert client.get("/api/evidence/"+eid).headers["content-type"].startswith("image/jpeg")
    analyze=client.post("/api/cases/"+ident+"/analyze/"+eid)
    assert analyze.status_code == 503
    assert "YOLO no configurado" in analyze.json()["detail"]

def test_production_admin_bootstrap():
    import subprocess, sys, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, APP_ENV="production", TASACION_DATA=tmp,
                   TASACION_DB=os.path.join(tmp,"cases.sqlite3"),
                   TASACION_ADMIN_PASSWORD="production-test-password",
                   SESSION_SECRET="production-only-test-secret-over-32-characters")
        run=subprocess.run(
            [sys.executable, "-c", "import sqlite3,app; con=sqlite3.connect(app.DB); assert con.execute(\"SELECT COUNT(*) FROM users WHERE username='administrador'\").fetchone()[0] == 1"],
            cwd=os.path.dirname(os.path.dirname(__file__)), env=env, capture_output=True, text=True,
        )
        assert run.returncode == 0, run.stderr
