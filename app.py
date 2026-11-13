import io
import json
import os
import secrets
import sqlite3
import hashlib
import hmac
import uuid
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DATA = Path(os.getenv("TASACION_DATA", str(ROOT / "data")))
DATA.mkdir(parents=True, exist_ok=True)
DB = Path(os.getenv("TASACION_DB", str(DATA / "cases.sqlite3")))
DB.parent.mkdir(parents=True, exist_ok=True)
UPLOADS = DATA / "evidence"
UPLOADS.mkdir(parents=True, exist_ok=True)
SESSION_SECRET = os.getenv("SESSION_SECRET", "")
ENV = os.getenv("APP_ENV", "development")
if ENV == "production" and (len(SESSION_SECRET) < 32 or not os.getenv("TASACION_ADMIN_PASSWORD")):
    raise RuntimeError("Production requires SESSION_SECRET (32+ chars) and TASACION_ADMIN_PASSWORD")
if not SESSION_SECRET:
    SESSION_SECRET = secrets.token_urlsafe(48)
app = FastAPI(title="Tasación Visual", version="1.0.0")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="lax", https_only=ENV == "production", max_age=8 * 3600)

def now():
    return datetime.now(timezone.utc).isoformat()

@contextmanager
def db():
    con = sqlite3.connect(DB, timeout=20)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()

def init_db():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS users (
          username TEXT PRIMARY KEY, salt TEXT NOT NULL, digest TEXT NOT NULL, role TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS cases (
          id TEXT PRIMARY KEY, asset_type TEXT NOT NULL, title TEXT NOT NULL,
          owner_name TEXT NOT NULL DEFAULT '', location TEXT NOT NULL DEFAULT '',
          details TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'abierto',
          base_value REAL NOT NULL DEFAULT 0, source TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS evidence (
          id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
          filename TEXT NOT NULL, original_name TEXT NOT NULL, shot TEXT NOT NULL,
          content_type TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS findings (
          id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
          evidence_id TEXT REFERENCES evidence(id) ON DELETE SET NULL, category TEXT NOT NULL,
          severity TEXT NOT NULL, confidence REAL, origin TEXT NOT NULL,
          state TEXT NOT NULL DEFAULT 'pendiente', description TEXT NOT NULL DEFAULT '',
          adjustment REAL NOT NULL DEFAULT 0, justification TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS audit (
          id INTEGER PRIMARY KEY AUTOINCREMENT, actor TEXT NOT NULL, action TEXT NOT NULL,
          case_id TEXT, at TEXT NOT NULL);
        """)
        for role in ("administrador", "revisor", "operador"):
            password = os.getenv("TASACION_" + role.upper() + "_PASSWORD")
            if role == "administrador" and not password and ENV != "production":
                password = os.getenv("TASACION_ADMIN_PASSWORD")
            if password:
                existing = con.execute("SELECT username FROM users WHERE username=?", (role,)).fetchone()
                if not existing:
                    salt = secrets.token_hex(16)
                    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 300000).hex()
                    con.execute("INSERT INTO users VALUES (?,?,?,?)", (role, salt, digest, role))

init_db()

def auth(req: Request, write=False, review=False):
    username = req.session.get("username")
    if not username:
        raise HTTPException(401, "Inicia sesión")
    if review and username not in ("administrador", "revisor"):
        raise HTTPException(403, "Se requiere rol revisor")
    if write and req.method in ("POST", "PATCH", "PUT", "DELETE"):
        origin = req.headers.get("origin")
        host = req.headers.get("host")
        if origin:
            from urllib.parse import urlparse
            parsed = urlparse(origin)
            if parsed.netloc != host or parsed.scheme != ("https" if ENV == "production" else parsed.scheme):
                raise HTTPException(403, "Origen no permitido")
    return username

def log(actor, action, case_id=None):
    with db() as con:
        con.execute("INSERT INTO audit(actor,action,case_id,at) VALUES(?,?,?,?)", (actor, action, case_id, now()))

def require_case(case_id):
    with db() as con:
        row = con.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Expediente no encontrado")
    return dict(row)

def serialize_case(row):
    c = dict(row)
    with db() as con:
        c["evidence"] = [dict(r) for r in con.execute("SELECT id,case_id,original_name,shot,content_type,created_at FROM evidence WHERE case_id=? ORDER BY created_at DESC", (c["id"],))]
        c["findings"] = [dict(r) for r in con.execute("SELECT * FROM findings WHERE case_id=? ORDER BY created_at DESC", (c["id"],))]
    accepted = [x for x in c["findings"] if x["state"] == "confirmado"]
    deductions = sum(max(0, float(x["adjustment"])) for x in accepted)
    c["estimated_value"] = round(max(0, float(c["base_value"]) - deductions), 2)
    c["unreviewed"] = sum(x["state"] == "pendiente" for x in c["findings"])
    c["valuation_status"] = "referencial" if c["base_value"] > 0 and not c["unreviewed"] and c["source"].strip() else "pendiente"
    return c

class Login(BaseModel):
    username: str
    password: str

class CaseIn(BaseModel):
    asset_type: str
    title: str = Field(min_length=2, max_length=140)
    owner_name: str = Field(default="", max_length=140)
    location: str = Field(default="", max_length=240)
    details: str = Field(default="", max_length=2000)

class BaseIn(BaseModel):
    base_value: float = Field(ge=0, le=1e12)
    source: str = Field(min_length=3, max_length=1000)

class FindingIn(BaseModel):
    category: str = Field(min_length=2, max_length=120)
    severity: str
    evidence_id: str | None = None
    description: str = Field(default="", max_length=2000)
    adjustment: float = Field(default=0, ge=0, le=1e12)
    justification: str = Field(default="", max_length=1000)

class ReviewIn(BaseModel):
    state: str
    adjustment: float = Field(default=0, ge=0, le=1e12)
    justification: str = Field(default="", max_length=1000)

@app.get("/health")
def health():
    return {"status": "ok", "service": "tasacion-visual", "model_enabled": bool(os.getenv("YOLO_WEIGHTS"))}

@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse(ROOT / "static" / "index.html")

@app.post("/api/login")
def login(data: Login, req: Request):
    with db() as con:
        row = con.execute("SELECT * FROM users WHERE username=?", (data.username,)).fetchone()
    if row:
        derived = hashlib.pbkdf2_hmac("sha256", data.password.encode(), bytes.fromhex(row["salt"]), 300000).hex()
        if hmac.compare_digest(derived, row["digest"]):
            req.session.clear()
            req.session["username"] = row["username"]
            return {"username": row["username"], "role": row["role"]}
    raise HTTPException(401, "Credenciales inválidas")

@app.post("/api/logout")
def logout(req: Request):
    req.session.clear()
    return {"ok": True}

@app.get("/api/me")
def me(req: Request):
    username = auth(req)
    return {"username": username, "role": username}

@app.get("/api/cases")
def list_cases(req: Request):
    auth(req)
    with db() as con:
        return [dict(row) for row in con.execute("SELECT * FROM cases ORDER BY updated_at DESC LIMIT 250")]

@app.post("/api/cases", status_code=201)
def create_case(data: CaseIn, req: Request):
    actor = auth(req, write=True)
    if data.asset_type not in ("vivienda", "mototaxi", "maquinaria", "terreno", "vehiculo"):
        raise HTTPException(422, "Tipo de activo inválido")
    case_id = str(uuid.uuid4())
    with db() as con:
        con.execute("INSERT INTO cases(id,asset_type,title,owner_name,location,details,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                    (case_id,data.asset_type,data.title,data.owner_name,data.location,data.details,now(),now()))
    log(actor, "crear_expediente", case_id)
    return serialize_case(require_case(case_id))

@app.get("/api/cases/{case_id}")
def get_case(case_id: str, req: Request):
    auth(req)
    return serialize_case(require_case(case_id))

@app.put("/api/cases/{case_id}/base")
def update_base(case_id: str, data: BaseIn, req: Request):
    actor = auth(req, write=True, review=True)
    require_case(case_id)
    with db() as con:
        con.execute("UPDATE cases SET base_value=?,source=?,updated_at=? WHERE id=?", (data.base_value,data.source,now(),case_id))
    log(actor, "actualizar_valor_base", case_id)
    return serialize_case(require_case(case_id))

@app.post("/api/cases/{case_id}/evidence", status_code=201)
async def upload_evidence(case_id: str, req: Request, shot: str = Form("general"), file: UploadFile = File(...)):
    actor = auth(req, write=True)
    require_case(case_id)
    if not file.content_type or file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(415, "Solo JPG, PNG o WebP")
    content = await file.read(8 * 1024 * 1024 + 1)
    if len(content) > 8 * 1024 * 1024:
        raise HTTPException(413, "La imagen excede 8 MB")
    try:
        im = Image.open(io.BytesIO(content))
        im.verify()
        im = Image.open(io.BytesIO(content))
        if im.width * im.height > 32_000_000:
            raise ValueError("Imagen demasiado grande")
        im = im.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(422, "Imagen inválida") from exc
    eid = str(uuid.uuid4())
    path = UPLOADS / (eid + ".jpg")
    im.thumbnail((2200, 2200))
    im.save(path, format="JPEG", quality=88)
    with db() as con:
        con.execute("INSERT INTO evidence VALUES(?,?,?,?,?,?,?)",
                    (eid,case_id,path.name,(file.filename or "foto")[:200],shot[:80],"image/jpeg",now()))
        con.execute("UPDATE cases SET updated_at=? WHERE id=?", (now(), case_id))
    log(actor, "subir_evidencia", case_id)
    return {"id": eid}

@app.get("/api/evidence/{evidence_id}")
def get_evidence(evidence_id: str, req: Request):
    auth(req)
    with db() as con:
        row = con.execute("SELECT filename FROM evidence WHERE id=?", (evidence_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Imagen no encontrada")
    return FileResponse(UPLOADS / row["filename"], media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})

@app.post("/api/cases/{case_id}/findings", status_code=201)
def create_finding(case_id: str, data: FindingIn, req: Request):
    actor = auth(req, write=True)
    require_case(case_id)
    if data.severity not in ("leve", "moderada", "severa"):
        raise HTTPException(422, "Severidad inválida")
    if data.evidence_id:
        with db() as con:
            hit = con.execute("SELECT 1 FROM evidence WHERE id=? AND case_id=?", (data.evidence_id,case_id)).fetchone()
        if not hit:
            raise HTTPException(422, "Evidencia no pertenece al expediente")
    fid = str(uuid.uuid4())
    with db() as con:
        con.execute("INSERT INTO findings(id,case_id,evidence_id,category,severity,confidence,origin,state,description,adjustment,justification,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (fid,case_id,data.evidence_id,data.category,data.severity,None,"manual","pendiente",data.description,data.adjustment,data.justification,now()))
    log(actor, "crear_hallazgo", case_id)
    return {"id": fid, "state": "pendiente"}

@app.put("/api/findings/{finding_id}/review")
def review_finding(finding_id: str, data: ReviewIn, req: Request):
    actor = auth(req, write=True, review=True)
    if data.state not in ("confirmado", "descartado", "pendiente"):
        raise HTTPException(422, "Estado inválido")
    if data.state == "confirmado" and data.adjustment > 0 and not data.justification.strip():
        raise HTTPException(422, "Documenta la justificación del ajuste")
    with db() as con:
        row = con.execute("SELECT case_id FROM findings WHERE id=?", (finding_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Hallazgo no encontrado")
        con.execute("UPDATE findings SET state=?,adjustment=?,justification=? WHERE id=?",
                    (data.state, data.adjustment if data.state == "confirmado" else 0,data.justification,finding_id))
        case_id = row["case_id"]
    log(actor, "revisar_hallazgo", case_id)
    return {"ok": True}

@app.post("/api/cases/{case_id}/analyze/{evidence_id}")
def analyze(case_id: str, evidence_id: str, req: Request):
    actor = auth(req, write=True)
    require_case(case_id)
    with db() as con:
        row = con.execute("SELECT * FROM evidence WHERE id=? AND case_id=?", (evidence_id,case_id)).fetchone()
    if not row:
        raise HTTPException(404, "Evidencia no encontrada")
    weights = os.getenv("YOLO_WEIGHTS")
    if not weights:
        raise HTTPException(503, "YOLO no configurado. Define YOLO_WEIGHTS con un modelo validado; no se simulan detecciones.")
    weights_path = Path(weights).resolve()
    if not weights_path.is_file():
        raise HTTPException(503, "No se encuentran los pesos YOLO")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise HTTPException(503, "Instala requirements-yolo.txt") from exc
    model = YOLO(str(weights_path))
    results = model.predict(str(UPLOADS / row["filename"]), conf=0.35, verbose=False)
    new_count = 0
    with db() as con:
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = str(result.names.get(cls, cls))
                con.execute("INSERT INTO findings(id,case_id,evidence_id,category,severity,confidence,origin,state,description,adjustment,justification,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            (str(uuid.uuid4()),case_id,evidence_id,label,"leve",conf,"yolo","pendiente","Detección automática: verificar con un revisor",0,"",now()))
                new_count += 1
    log(actor, "analisis_yolo", case_id)
    return {"detections": new_count, "warning": "Detecciones pendientes; no generan descuento automático. Un detector genérico no certifica daños."}

@app.get("/api/cases/{case_id}/report.json")
def report_json(case_id: str, req: Request):
    auth(req)
    case = serialize_case(require_case(case_id))
    log(req.session["username"],"exportar_json",case_id)
    return JSONResponse(case, headers={"Content-Disposition": 'attachment; filename="tasacion.json"'})

@app.get("/api/cases/{case_id}/report.pdf")
def report_pdf(case_id: str, req: Request):
    auth(req)
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen.canvas import Canvas
    case = serialize_case(require_case(case_id))
    buf = io.BytesIO()
    pdf = Canvas(buf, pagesize=A4)
    pdf.setTitle("Informe de tasación referencial")
    lines = ["INFORME DE TASACION REFERENCIAL",
             "Expediente: " + case["id"], "Activo: " + case["asset_type"],
             "Descripcion: " + case["title"][:90],
             "Ubicacion: " + case["location"][:90],
             "Valor base: S/ " + format(case["base_value"],",.2f"),
             "Valor estimado: S/ " + format(case["estimated_value"],",.2f"),
             "Estado: " + case["valuation_status"],
             "Fuente: " + case["source"][:100], "",
             "HALLAZGOS"]
    for f in case["findings"]:
        lines.append((f["category"] + " | " + f["severity"] + " | " + f["state"] + " | Ajuste S/ " + str(f["adjustment"]))[:105])
    lines += ["", "Estimacion no constituye peritaje, credito aprobado ni diagnostico estructural."]
    y = 800
    for line in lines:
        if y < 55:
            pdf.showPage()
            y = 800
        pdf.drawString(40,y,line)
        y -= 19
    pdf.save()
    log(req.session["username"],"exportar_pdf",case_id)
    return Response(buf.getvalue(), media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="tasacion.pdf"'})

@app.get("/api/cases/{case_id}/audit")
def audit(case_id: str, req: Request):
    auth(req, review=True)
    require_case(case_id)
    with db() as con:
        return [dict(r) for r in con.execute("SELECT * FROM audit WHERE case_id=? ORDER BY id DESC LIMIT 100",(case_id,))]
