"""Administrator backup protects cases and photos on ephemeral deployments."""
import io
import json
import sqlite3
import zipfile

from PIL import Image
from fastapi.testclient import TestClient

# This test module is collected alphabetically before test_app.py; configure
# the same isolated admin identity before importing app (which calls init_db).
import os
os.environ["APP_ENV"]="development"
os.environ["TASACION_ADMIN_PASSWORD"]="strong-test-password"
os.environ["SESSION_SECRET"]="local-testing-secret-do-not-use-in-production"

import app


def test_admin_backup_is_authenticated_and_contains_database_snapshot():
    unauth=TestClient(app.app).get("/api/admin/backup")
    assert unauth.status_code==401
    client=TestClient(app.app)
    logged=client.post("/api/login",json={
        "username":"administrador","password":"strong-test-password"})
    assert logged.status_code==200
    case=client.post("/api/cases",json={"title":"Control de respaldo","asset_type":"vivienda"})
    assert case.status_code==201,case.text
    cid=case.json()["id"]
    photo=io.BytesIO()
    Image.new("RGB",(64,48),"white").save(photo,format="JPEG")
    saved=client.post(f"/api/cases/{cid}/evidence",files={
        "file":("pared.jpg",photo.getvalue(),"image/jpeg")},data={"shot":"Interior"})
    assert saved.status_code==201,saved.text
    backup=client.get("/api/admin/backup")
    assert backup.status_code==200,backup.text[:150]
    assert backup.headers["content-type"]=="application/zip"
    assert backup.headers["cache-control"]=="private, no-store"
    with zipfile.ZipFile(io.BytesIO(backup.content)) as archive:
        names=archive.namelist()
        assert "cases.sqlite3" in names
        assert "manifest.json" in names
        assert any(name.startswith("evidence/") and name.endswith(".jpg") for name in names)
        manifest=json.loads(archive.read("manifest.json"))
        assert manifest["contains_private_information"] is True
        database=io.BytesIO(archive.read("cases.sqlite3"))
        # A snapshot of SQLite is readable independently from the running database.
        assert database.getvalue().startswith(b"SQLite format 3")
        assert cid.encode() in database.getvalue()


def test_reviewer_cannot_download_private_admin_backup(monkeypatch):
    import uuid
    username="reviewer_backup_"+uuid.uuid4().hex[:10]
    password="reviewer-backup-test"
    import secrets,hashlib
    salt=secrets.token_hex(16)
    digest=hashlib.pbkdf2_hmac("sha256",password.encode(),bytes.fromhex(salt),300000).hex()
    with app.db() as con:
        con.execute("INSERT INTO users(username,salt,digest,role) VALUES(?,?,?,?)",
                    (username,salt,digest,"revisor"))
    client=TestClient(app.app)
    assert client.post("/api/login",json={"username":username,"password":password}).status_code==200
    response=client.get("/api/admin/backup")
    assert response.status_code==403
