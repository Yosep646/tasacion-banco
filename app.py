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
from yolo_service import predict_findings, get_model_configuration
from vision_service import analyze_photo, evaluate_condition, guided_shots, simulate_credit
from yolo_onnx import predict_cracks
from yolo_defects import detect_defects
from yolo_paint import detect_paint
from surface_segmentation import segment_surfaces, model_path as surface_model_path, MODEL_SHA256 as SURFACE_MODEL_SHA256
from inspection_taxonomy import public_capabilities, improvement_for
from valuation_reference import estimate_property_scenario
from official_reference import official_unit_values, reference_scenario
from rapid_service import summarize
from repair_pricing import build_automatic_quote
from starlette.concurrency import run_in_threadpool

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from PIL import Image, ImageDraw, UnidentifiedImageError
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
        CREATE TABLE IF NOT EXISTS visual_assessments (
          id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
          evidence_id TEXT NOT NULL REFERENCES evidence(id) ON DELETE CASCADE,
          state TEXT NOT NULL DEFAULT 'pendiente', result_json TEXT NOT NULL,
          reviewer TEXT, reviewed_at TEXT, created_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS visual_assessments_case_idx ON visual_assessments(case_id,evidence_id,created_at);
        CREATE TABLE IF NOT EXISTS rapid_reports (\n          case_id TEXT PRIMARY KEY REFERENCES cases(id) ON DELETE CASCADE,\n          result_json TEXT NOT NULL, created_at TEXT NOT NULL);\n        CREATE TABLE IF NOT EXISTS audit (
          id INTEGER PRIMARY KEY AUTOINCREMENT, actor TEXT NOT NULL, action TEXT NOT NULL,
          case_id TEXT, at TEXT NOT NULL);
        """)
        for role in ("administrador", "revisor", "operador"):
            password = os.getenv("TASACION_" + role.upper() + "_PASSWORD")
            if role == "administrador" and not password:
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
        c["visual_assessments"] = [dict(r) for r in con.execute("SELECT * FROM visual_assessments WHERE case_id=? ORDER BY created_at DESC, rowid DESC", (c["id"],))]
        quick = con.execute("SELECT result_json FROM rapid_reports WHERE case_id=?", (c["id"],)).fetchone()
    c["quick_report"] = json.loads(quick["result_json"]) if quick else None
    if c["quick_report"] is not None:
        q = c["quick_report"]
        q.setdefault("automatic_repair_scenario",build_automatic_quote(q.get("issues",[])))
        q.setdefault("illustrative_construction_example",reference_scenario(60,"sierra","D",illustrative=True))
        indexed={(x.get("evidence_id"),x.get("category"))
                 for x in c["findings"] if x.get("origin")=="quick_ai"}
        c["quick_findings_need_sync"] = any(
            (x.get("evidence_id"),x.get("category")) not in indexed
            for x in q.get("issues",[]) if isinstance(x,dict) and
            x.get("evidence_id") and x.get("category")
        )
    else:
        c["quick_findings_need_sync"] = False
    c["visual_condition"] = evaluate_condition(c["asset_type"], c["evidence"], c["visual_assessments"], c["findings"])
    accepted = [x for x in c["findings"] if x["state"] == "confirmado"]
    deductions = sum(max(0, float(x["adjustment"])) for x in accepted)
    c["estimated_value"] = round(max(0, float(c["base_value"]) - deductions), 2)
    c["unreviewed"] = sum(x["state"] == "pendiente" for x in c["findings"])
    c["valuation_status"] = "referencial" if c["base_value"] > 0 and not c["unreviewed"] and c["source"].strip() else "pendiente"
    return c

class VisualReview(BaseModel):
    state: str

class ManualVisualObservation(BaseModel):
    condition: str
    category: str = Field(default="", max_length=120)
    severity: str = "leve"
    description: str = Field(default="", max_length=500)
    improvement: str = Field(default="", max_length=300)
    material: str = Field(default="", max_length=100)
    notes: str = Field(default="", max_length=350)

class PropertyScenario(BaseModel):
    built_area_m2: float = Field(gt=0, le=100000)
    construction_unit_pen_m2: float = Field(gt=0, le=100000)
    source: str = Field(min_length=8,max_length=500)
    land_area_m2: float = Field(default=0,ge=0,le=100000)
    land_unit_pen_m2: float = Field(default=0,ge=0,le=1000000)
    land_source: str = Field(default="",max_length=500)

class OfficialConstructionReference(BaseModel):
    built_area_m2: float = Field(gt=0, le=50000)
    zone: str
    grade: str

class CreditSimulation(BaseModel):
    documented_budget: float = Field(gt=0, le=1e10)
    budget_source: str = Field(min_length=5, max_length=500)

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
    return {"status": "ok", "service": "tasacion-visual", "models": {"vivienda_fisuras": bool(os.getenv("YOLO_VIVIENDA_WEIGHTS")), "generico": bool(os.getenv("YOLO_WEIGHTS"))}, "vision_enabled": bool(os.getenv("OPENAI_API_KEY")), "yolo_cpu_enabled": os.getenv("YOLO_CPU_ENABLED") == "1", "multiclass_yolo_enabled": os.getenv("YOLO_DEFECT5_ENABLED") == "1", "surface_segmentation_enabled": os.getenv("SURFACE_SEGMENTATION_ENABLED") == "1", "paint_experimental_enabled": os.getenv("YOLO_PAINT_ENABLED") == "1" and len(os.getenv("YOLO_PAINT_SHA256","")) == 64}

def _model_cache_status(file_path, expected_sha256=None):
    """Read-only readiness: never download model weights from a health check."""
    try:
        candidate=Path(file_path)
        if not candidate.is_file():
            return "awaiting_first_inference"
        if expected_sha256:
            digest=hashlib.sha256(candidate.read_bytes()).hexdigest()
            return "cached_verified" if hmac.compare_digest(digest, expected_sha256) else "checksum_mismatch"
        return "cached_unverified"
    except (OSError,ValueError):
        return "cache_unreadable"


@app.get("/api/model-readiness")
def model_readiness(req: Request):
    auth(req)
    from yolo_defects import model_path as defect5_model_path, PINNED_SHA256
    from yolo_onnx import model_dir as crack_model_dir, NAME as CRACK_MODEL
    surfaces_enabled=os.getenv("SURFACE_SEGMENTATION_ENABLED")=="1"
    defect5_enabled=(os.getenv("YOLO_CPU_ENABLED")=="1" and os.getenv("YOLO_DEFECT5_ENABLED")=="1")
    paint_sha=os.getenv("YOLO_PAINT_SHA256","").lower()
    paint_enabled=(os.getenv("YOLO_CPU_ENABLED")=="1" and os.getenv("YOLO_PAINT_ENABLED")=="1"
                   and len(paint_sha)==64 and all(c in "0123456789abcdef" for c in paint_sha))
    return {
        "surface_segmentation":{"configured":surfaces_enabled,"state":_model_cache_status(surface_model_path(),SURFACE_MODEL_SHA256) if surfaces_enabled else "disabled","classes":["pared","piso","techo"],"model":"SegFormer-B0 ADE20K ONNX INT8","note":"Áreas de píxeles, no m² físicos; requiere revisión visual."},
        "yolo_defect5":{
            "configured":defect5_enabled,
            "state":_model_cache_status(defect5_model_path(),PINNED_SHA256) if defect5_enabled else "disabled",
            "classes":["fisura_visible","grieta_escalonada","moho_visible","pintura_desprendida","humedad_visible"],
            "note":"Configuración no implica inferencia exitosa. Se valida al analizar la primera foto.",
        },
        "yolo_cracks":{
            "configured":os.getenv("YOLO_CPU_ENABLED")=="1",
            "state":_model_cache_status(crack_model_dir()/CRACK_MODEL) if os.getenv("YOLO_CPU_ENABLED")=="1" else "disabled",
            "segmentation":"instance_masks_when_prototype_output_available",
            "note":"YOLOv8-seg delimita posibles fisuras por píxel si el ONNX incluye prototipos. No produce metrado real.",
        },
        "yolo_paint":{
            "configured":paint_enabled,
            "state":_model_cache_status(
                Path(os.getenv("TASACION_MODEL_DIR","/tmp/tasacion-models"))/"pintura-deteriorada-candidate.onnx",
                paint_sha) if paint_enabled else "disabled",
            "gate":"experimental_not_field_validated",
        },
        "excluded_classes":["piso_roto","puerta_visible","hueco_sin_puerta","pared_sin_tarrajeo",
                            "pared_tarrajeada","ladrillo_expuesto","calamina_visible","techo_concreto_visible"],
        "note":"Nunca interpretar ausencia de una detección como confirmación de buen estado.",
    }


@app.get("/api/model-capabilities")
def model_capabilities(req: Request):
    auth(req)
    return public_capabilities(os.getenv("YOLO_DEFECT5_ENABLED") == "1")

@app.get("/assets/hero-house.webp")
def hero_house_asset():
    return FileResponse(ROOT / "static" / "hero-house.webp", media_type="image/webp", headers={"Cache-Control":"public, max-age=86400"})

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
    asset_type = require_case(case_id)["asset_type"]
    model_config = get_model_configuration(asset_type)
    if model_config is None:
        raise HTTPException(503, "Modelo YOLO no configurado para este activo. Configura YOLO_VIVIENDA_WEIGHTS o YOLO_WEIGHTS.")
    try:
        proposed = predict_findings(str(UPLOADS / row["filename"]), model_config)
    except (ImportError, FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(503, "Inferencia YOLO no disponible: " + str(exc)) from exc
    new_count = 0
    with db() as con:
        for detection in proposed:
            con.execute("INSERT INTO findings(id,case_id,evidence_id,category,severity,confidence,origin,state,description,adjustment,justification,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), case_id, evidence_id, detection["category"], "leve",
                 detection["confidence"], detection["origin"], "pendiente",
                 detection["description"], 0, "", now()))
            new_count += 1
    log(actor, "analisis_yolo", case_id)
    return {"detections": new_count, "profile": model_config["profile"], "warning": "Detecciones pendientes; no generan descuento automático. Las fisuras son observaciones visuales, no diagnósticos estructurales."}


def _sync_quick_findings(con, case_id: str, summary: dict) -> int:
    """Materialize AI suggestions as pending findings without assigning severity.

    A maximum of one finding per image/category is created; repeated use is
    idempotent. Manual findings and reviewer decisions are NEVER overwritten.
    """
    candidates = {}
    for item in summary.get("issues", []):
        if not isinstance(item, dict):
            continue
        eid = str(item.get("evidence_id", ""))
        category = str(item.get("category", ""))[:120].strip()
        if not eid or not category:
            continue
        confidence = item.get("confidence")
        if isinstance(confidence, bool):
            confidence = None
        try:
            confidence = float(confidence) if confidence is not None else None
            if confidence is not None and (not 0 <= confidence <= 1):
                confidence = None
        except (ValueError, TypeError):
            confidence = None
        key = (eid, category)
        current = candidates.get(key)
        if current is None or (confidence if confidence is not None else -1) > current[0]:
            candidates[key] = (confidence if confidence is not None else -1, item)
    added = 0
    for (eid, category), (confidence, item) in candidates.items():
        if not con.execute("SELECT 1 FROM evidence WHERE id=? AND case_id=?", (eid, case_id)).fetchone():
            continue
        existing = con.execute(
            "SELECT 1 FROM findings WHERE case_id=? AND evidence_id=? AND category=? "
            "AND origin='quick_ai' LIMIT 1", (case_id, eid, category)
        ).fetchone()
        if existing:
            continue
        # No severity, deduction or monetary adjustment can be inferred from a box.
        con.execute(
            "INSERT INTO findings(id,case_id,evidence_id,category,severity,confidence,"
            "origin,state,description,adjustment,justification,created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()),case_id,eid,category,"no_determinada",
             None if confidence < 0 else confidence,"quick_ai","pendiente",
             str(item.get("description",""))[:2000],0,"",now()))
        added += 1
    return added


@app.post("/api/cases/{case_id}/sync-quick-report")
def sync_quick_report(case_id: str, req: Request):
    actor = auth(req, write=True)
    require_case(case_id)
    with db() as con:
        row = con.execute(
            "SELECT result_json FROM rapid_reports WHERE case_id=?", (case_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404,"Este expediente no tiene evaluación rápida")
        count = _sync_quick_findings(con, case_id, json.loads(row["result_json"]))
    if count:
        log(actor,"sincronizar_hallazgos_automaticos",case_id)
    return {"created":count,"case_id":case_id,"status":"pendiente_de_revision"}


async def _save_quick_photo(case_id: str, file: UploadFile, shot: str, actor: str):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(415, "Se requiere una fotografía JPG, PNG o WebP")
    content = await file.read(8 * 1024 * 1024 + 1)
    if len(content) > 8 * 1024 * 1024:
        raise HTTPException(413, "Cada imagen debe pesar menos de 8 MB")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            if image.width * image.height > 32_000_000:
                raise ValueError("Imagen demasiado grande")
            processed = image.convert("RGB")
            processed.thumbnail((1600, 1600))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(422, "Imagen inválida") from exc
    eid = str(uuid.uuid4())
    filename = eid + ".jpg"
    processed.save(UPLOADS / filename, format="JPEG", quality=85)
    with db() as con:
        con.execute("INSERT INTO evidence VALUES(?,?,?,?,?,?,?)",
                    (eid,case_id,filename,(file.filename or "imagen")[:200],
                     shot,"image/jpeg",now()))
    log(actor,"subir_foto_rapida",case_id)
    return {"id":eid, "shot":shot,"filename":filename}


@app.post("/api/quick-inspection", status_code=201)
async def quick_inspection(req: Request, exterior: UploadFile = File(...),
                           interior: UploadFile | None = File(default=None)):
    actor = auth(req, write=True)
    can_yolo = os.getenv("YOLO_CPU_ENABLED") == "1"
    can_vision = bool(os.getenv("OPENAI_API_KEY"))
    can_surfaces = os.getenv("SURFACE_SEGMENTATION_ENABLED") == "1"
    if not can_yolo and not can_vision and not can_surfaces:
        raise HTTPException(503, "No hay un modelo de visión configurado para analizar fotografías")
    case_id = str(uuid.uuid4())
    title = "Inspección rápida " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with db() as con:
        con.execute("INSERT INTO cases(id,asset_type,title,created_at,updated_at) VALUES(?,?,?,?,?)",
                    (case_id,"vivienda",title,now(),now()))
    photos = []
    try:
        photos.append(await _save_quick_photo(case_id,exterior,"Exterior",actor))
        if interior is not None and interior.filename:
            photos.append(await _save_quick_photo(case_id,interior,"Interior",actor))
    except HTTPException:
        with db() as con:
            con.execute("DELETE FROM cases WHERE id=?",(case_id,))
        for p in photos:
            (UPLOADS / p["filename"]).unlink(missing_ok=True)
        raise
    analyses = []
    for photo in photos:
        result = None
        error = None
        annotated = None
        image_path = UPLOADS / photo["filename"]
        # YOLO detects possible cracks; multimodal model can add further visual observations.
        if can_vision:
            try:
                result = await run_in_threadpool(analyze_photo,str(image_path),photo["shot"],"vivienda")
                result["method"] = "multimodal_visual"
            except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
                error = "El proveedor multimodal no respondió correctamente"
        if can_yolo:
            try:
                if os.getenv("YOLO_DEFECT5_ENABLED") == "1":
                    defects, annotated = await run_in_threadpool(detect_defects, str(image_path))
                    method = "yolo26_defect5"
                else:
                    defects, annotated = await run_in_threadpool(predict_cracks, str(image_path))
                    method = "yolo_cpu_onnx"
                if annotated is not None:
                    annotated.save(UPLOADS / (photo["id"]+"-yolo.jpg"),format="JPEG",quality=85)
                if result is None:
                    result={
                        "condition":"requiere_mejoras" if defects else "no_determinable",
                        "summary":("Se identificaron "+str(len(defects))+" posible(s) anomalía(s) visible(s) que requieren revisión."
                                   if defects else "No se detectaron anomalías sobre el umbral. Esto NO acredita que la vivienda esté en buen estado."),
                        "materials":[], "issues":[], "limitations":[],
                        "method":method,
                    }
                for defect in defects:
                    result["issues"].append({
                        "category":defect["category"],
                        "severity":"no_determinada",
                        "description":defect["description"]+" (confianza "+str(round(defect["confidence"]*100))+"%)",
                        "improvement":improvement_for(defect["category"]),
                        "confidence":defect["confidence"],
                        "box":defect["box"],
                        "has_mask":defect.get("has_mask",False),
                        "mask_coverage_percent_image":defect.get("mask_coverage_percent_image"),
                        "origin":method,
                    })
                if defects:
                    result["condition"]="requiere_mejoras"
                if method == "yolo_cpu_onnx":
                    result["limitations"].append("Detector YOLO actual: solo posibles fisuras, no otros daños ni materiales.")
                else:
                    result["limitations"].append(
                        "YOLO de cinco defectos: fisuras, grietas escalonadas, moho, pintura desprendida y manchas de filtración. "
                        "El modelo tiene bajo desempeño publicado en pintura/humedad; confirmar siempre.")
                result["yolo_detections"]=len(defects)
                result["annotated_url"]="/api/evidence/"+photo["id"]+"/yolo-overlay"
                if result["method"]=="multimodal_visual":
                    result["method"]="multimodal_visual+"+method
            except (OSError,ValueError,RuntimeError,ImportError,IndexError,TypeError) as exc:
                error=(error+"; " if error else "")+"No fue posible ejecutar YOLO multiclase"
                # If the extended model cannot load, try the already tested crack model.
                if os.getenv("YOLO_DEFECT5_ENABLED")=="1":
                    try:
                        cracks, annotated=await run_in_threadpool(predict_cracks,str(image_path))
                        annotated.save(UPLOADS/(photo["id"]+"-yolo.jpg"),format="JPEG",quality=85)
                        if result is None:
                            result={"condition":"requiere_mejoras" if cracks else "no_determinable",
                                "summary":"Se utilizó el detector de respaldo de fisuras.",
                                "materials":[],"issues":[],"limitations":[],"method":"yolo_cpu_onnx_fallback"}
                        result["issues"].extend([{
                            "category":c["category"],"severity":"no_determinada",
                            "description":c["description"],"improvement":improvement_for(c["category"]),
                            "confidence":c["confidence"],"box":c["box"],
                            "has_mask":c.get("has_mask",False),
                            "mask_coverage_percent_image":c.get("mask_coverage_percent_image"),
                            "origin":"yolo_cpu_onnx_fallback",
                        } for c in cracks])
                        if cracks:result["condition"]="requiere_mejoras"
                        result["annotated_url"]="/api/evidence/"+photo["id"]+"/yolo-overlay"
                        result["limitations"].append("Modelo multiclase no disponible: se aplicó solo detector de fisuras.")
                    except (OSError,ValueError,RuntimeError,ImportError):
                        pass
        # Optional specialized paint model. Strictly gated by a pinned checksum and
        # deliberately disabled until its real holdout metrics meet acceptance criteria.
        if can_yolo and os.getenv("YOLO_PAINT_ENABLED")=="1":
            try:
                paint_issues, paint_overlay = await run_in_threadpool(detect_paint,str(image_path))
                if result is None:
                    result={
                        "condition":"no_determinable",
                        "summary":"Análisis exploratorio de pintura completado; resultados no validados en campo.",
                        "materials":[],"issues":[],"limitations":[],
                        "method":"yolo_paint_experimental",
                    }
                if paint_issues:
                    for issue in paint_issues:
                        result.setdefault("experimental_issues",[]).append({
                            "category":issue["category"],
                            "severity":"no_determinada",
                            "description":issue["description"]+" (confianza "+str(round(issue["confidence"]*100))+"%)",
                            "improvement":issue["improvement"],
                            "confidence":issue["confidence"],
                            "box":issue["box"],
                            "origin":"yolo_paint_experimental",
                        })
                    if "yolo_paint_experimental" not in result["method"]:
                        result["method"] += "+yolo_paint_experimental"
                overlay_file=UPLOADS/(photo["id"]+"-yolo.jpg")
                if overlay_file.exists():
                    with Image.open(overlay_file) as existing:
                        combined=existing.convert("RGB")
                    drawer=ImageDraw.Draw(combined)
                    for issue in paint_issues:
                        box=tuple(issue["box"])
                        drawer.rectangle(box,outline="#e69922",width=max(2,int(combined.width/350)))
                        drawer.text((box[0],max(0,box[1]-14)),"Pintura "+str(round(issue["confidence"]*100))+"%",fill="#b56a13")
                    combined.save(overlay_file,format="JPEG",quality=85)
                else:
                    paint_overlay.save(overlay_file,format="JPEG",quality=85)
                result["annotated_url"]="/api/evidence/"+photo["id"]+"/yolo-overlay"
                result["limitations"].append(
                    "Modelo experimental de deterioro de pintura: requiere verificación "
                    "en viviendas reales de Perú y no debe usarse para aprobar créditos.")
            except (OSError,ValueError,RuntimeError,ImportError,IndexError,TypeError) as exc:
                error=(error+"; " if error else "")+"El detector experimental de pintura no estuvo disponible"
        if result is None:
            result={"condition":"no_determinable","summary":"No se pudo completar la inferencia visual.",
                    "materials":[],"issues":[],"limitations":["No hay resultados de IA verificados."],
                    "method":"inferencia_fallida"}
        aid = str(uuid.uuid4())
        with db() as con:
            con.execute("INSERT INTO visual_assessments(id,case_id,evidence_id,state,result_json,created_at) VALUES(?,?,?,?,?,?)",
                        (aid,case_id,photo["id"],"pendiente",
                         json.dumps(result,ensure_ascii=False),now()))
        analyses.append({"evidence_id":photo["id"],"shot":photo["shot"],
                         "result":result,"error":error})
    summary=summarize(analyses,len(photos),can_vision)
    summary["case_id"]=case_id
    summary["generated_at"]=now()
    if all(a["result"]["method"]=="inferencia_fallida" for a in analyses):
        summary["status"]="analisis_no_disponible"
    with db() as con:
        con.execute("INSERT INTO rapid_reports VALUES(?,?,?)",
                    (case_id,json.dumps(summary,ensure_ascii=False),now()))
        _sync_quick_findings(con,case_id,summary)
    log(actor,"generar_informe_rapido",case_id)
    return summary


def _get_rapid_report(case_id: str, req: Request):
    auth(req)
    require_case(case_id)
    with db() as con:
        report = con.execute("SELECT result_json FROM rapid_reports WHERE case_id=?",
                             (case_id,)).fetchone()
    if not report:
        raise HTTPException(404, "Este expediente no tiene informe rápido")
    result=json.loads(report["result_json"])
    result.setdefault("illustrative_construction_example",reference_scenario(60,"sierra","D",illustrative=True))
    result.setdefault("automatic_repair_scenario",build_automatic_quote(result.get("issues",[])))
    return result


@app.get("/api/cases/{case_id}/quick-report")
def quick_report(case_id: str, req: Request):
    return _get_rapid_report(case_id,req)


@app.get("/api/official-construction-values")
def get_official_construction_values(req: Request):
    auth(req)
    return official_unit_values()


@app.post("/api/cases/{case_id}/official-construction-reference")
def set_official_construction_reference(case_id: str, data: OfficialConstructionReference, req: Request):
    actor=auth(req,write=True)
    require_case(case_id)
    with db() as con:
        row=con.execute("SELECT result_json FROM rapid_reports WHERE case_id=?",(case_id,)).fetchone()
    if not row:
        raise HTTPException(404,"Primero genera el informe de inspección rápida")
    try:
        calculation=reference_scenario(data.built_area_m2,data.zone,data.grade)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    report=json.loads(row["result_json"])
    report["official_construction_reference"]=calculation
    with db() as con:
        con.execute("UPDATE rapid_reports SET result_json=? WHERE case_id=?",
                    (json.dumps(report,ensure_ascii=False),case_id))
    log(actor,"registrar_valor_unitario_oficial_edificacion",case_id)
    return calculation


@app.post("/api/cases/{case_id}/valuation-scenario")
def valuation_scenario(case_id: str, data: PropertyScenario, req: Request):
    actor=auth(req,write=True,review=True)
    require_case(case_id)
    with db() as con:
        report=con.execute("SELECT result_json FROM rapid_reports WHERE case_id=?",(case_id,)).fetchone()
    if not report:
        raise HTTPException(404,"Primero ejecuta una inspección rápida")
    try:
        result=estimate_property_scenario(
            data.built_area_m2,data.construction_unit_pen_m2,data.source,
            data.land_area_m2,data.land_unit_pen_m2,data.land_source)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    stored=json.loads(report["result_json"])
    stored["documented_valuation_scenario"]=result
    with db() as con:
        con.execute("UPDATE rapid_reports SET result_json=? WHERE case_id=?",
                    (json.dumps(stored,ensure_ascii=False),case_id))
    log(actor,"registrar_escenario_valor_referencial",case_id)
    return result


@app.get("/api/cases/{case_id}/quick-report.pdf")
def quick_report_pdf(case_id: str, req: Request):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import simpleSplit
    from reportlab.pdfgen.canvas import Canvas
    report = _get_rapid_report(case_id,req)
    stream = io.BytesIO()
    pdf = Canvas(stream,pagesize=A4)
    pdf.setTitle("Informe inmediato de inspección visual preliminar")
    y = 794

    def draw(value, emphasize=False):
        nonlocal y
        pdf.setFont("Helvetica-Bold" if emphasize else "Helvetica", 11 if emphasize else 9)
        safe_text = str(value).encode("latin-1","replace").decode("latin-1")
        for line in simpleSplit(safe_text,"Helvetica",9,520):
            if y < 55:
                pdf.showPage()
                y=790
            pdf.drawString(38,y,line)
            y -= 15
    draw("INFORME VISUAL PRELIMINAR - NO ES TASACIÓN BANCARIA",True)
    draw("Expediente: "+case_id)
    draw("Fecha UTC: "+report["generated_at"])
    draw("Fotografías analizadas: "+str(report["images_analyzed"]))
    draw("Estado: "+report["condition"],True)
    draw("Métodos: "+", ".join(report["models"]))
    for item in report["image_summaries"]:
        draw(item["shot"]+": "+str(item["summary"]))
    draw("PLAN DE MEJORAS E INSPECCIONES",True)
    if not report.get("improvements"):
        draw("Sin intervenciones confirmadas; revisar otros ambientes.")
    for item in report.get("improvements",[]):
        draw(item["shot"]+" | "+item["label"]+" | prioridad "+item["priority"])
        draw(item["action"])
        draw(item["before_quoting"])
    draw("No se verificaron metrados ni causas estructurales.")
    draw("CATEGORÍAS NO EVALUADAS POR EL MODELO",True)
    for category in report.get("capabilities",{}).get("training_required",[]):
        draw(category)
    draw("MATERIALES OBSERVADOS",True)
    if not report["materials"]:
        draw("No identificables con los modelos disponibles")
    for material in report["materials"]:
        draw(material["name"]+" - "+material["location"])
    draw("POSIBLES HALLAZGOS",True)
    for item in report["issues"]:
        draw(item["category"]+": "+item["description"])
        draw("Mejora posible: "+item["improvement"])
    if not report["issues"]:
        draw("No se identificaron hallazgos; esto no certifica buen estado.")
    if report.get("experimental_issues"):
        draw("ALERTAS EXPERIMENTALES - EXCLUIDAS DE LA TASACIÓN",True)
        for item in report["experimental_issues"]:
            draw(item["shot"]+" - "+item["label"]+" (requiere revisión profesional)")
            draw(item["improvement"])
        draw("Alertas sin impacto automático en condición, costos o valoración.")
    construction=report.get("official_construction_reference")
    if construction:
        draw("REFERENCIA DE EDIFICACIÓN SEGÚN TABLA OFICIAL 2026",True)
        draw("Zona: "+construction["zone"]+" | Categoría uniforme declarada: "+construction["grade"])
        draw("Área techada declarada: "+str(construction["built_area_m2"])+" m2")
        draw("Valor unitario referencial: S/ "+str(construction["unit_cost_pen_m2"])+" por m2")
        draw("Referencia de edificación: S/ "+str(construction["construction_reference_pen"]),True)
        draw("Fuente: "+construction["source_url"])
        draw("SIN TERRENO, DEPRECIACIÓN NI VALOR COMERCIAL; categorías no verificadas por fotos.")
    else:
        example=reference_scenario(60,"sierra","D",illustrative=True)
        draw("EJEMPLO DIDÁCTICO, NO VALOR DE LA VIVIENDA",True)
        draw("Sierra, 60 m2, categoría D hipotética: S/ "+str(example["construction_reference_pen"]))
        draw("Completa área real, zona y categoría para guardar una referencia de edificación.")
    draw("ESCENARIO AUTOMÁTICO ILUSTRATIVO DE REPARACIONES",True)
    provisional=report.get("automatic_repair_scenario") or build_automatic_quote(report.get("issues",[]))
    if provisional["illustrative_min_pen"] is not None:
        draw("Rango DIDÁCTICO, NO COTIZACIÓN FINAL: S/ "+str(provisional["illustrative_min_pen"])+" a S/ "+str(provisional["illustrative_max_pen"]),True)
    else:
        draw("Sin importe sustentable en las detecciones actuales.")
    draw(provisional["explanation"])
    for line in provisional["items"]:
        draw(line["shot"]+" | "+line["work"])
        draw("Tarifa: S/ "+str(line["unit_price_pen"])+"/"+line["unit"]+"; supuesto NO MEDIDO 1 a 3 "+line["unit"])
        draw("Ejemplo: S/ "+str(line["illustrative_min_pen"])+" a S/ "+str(line["illustrative_max_pen"]))
        draw("Fuente: "+line["source_url"])
        draw("Advertencia: "+line["note"])
    for line in provisional["excluded"]:
        draw("EXCLUIDO: "+line["category"]+" ("+line["shot"]+"): "+line["reason"])
    draw("COTIZACIÓN DEFINITIVA",True)
    draw(report["quote"]["explanation"])
    for item in report["quote"]["items"]:
        draw(item["work"]+": S/ "+str(item["unit_price_pen"])+"/"+item["unit"]+"; falta metrado.")
    draw("Total: pendiente de cuantificación. Valor comercial: no estimado.")
    draw(report["property_market_value_reason"])
    scenario=report.get("documented_valuation_scenario")
    if scenario:
        draw("ESCENARIO MONETARIO CON FUENTES DOCUMENTADAS",True)
        draw("Construcción: S/ "+str(scenario["construction_value_pen"]))
        draw("Terreno: S/ "+str(scenario["land_value_pen"]))
        draw("Total aritmético referencial: S/ "+str(scenario["indicative_total_pen"]))
        draw("Fuente construcción: "+scenario["construction_source"])
        if scenario["land_source"]:
            draw("Fuente terreno: "+scenario["land_source"])
        draw("No es una tasación comercial ni una garantía aprobada.")

    draw("LIMITACIONES Y VALIDACIÓN",True)
    for line in report["limitations"]:
        draw(line)
    draw("Informe automático orientativo. Requiere evaluación profesional.",True)
    pdf.save()
    return Response(stream.getvalue(),media_type="application/pdf",
        headers={"Content-Disposition":'attachment; filename="informe-visual-rapido.pdf"'})


@app.get("/api/inspection-guide/{asset_type}")
def inspection_guide(asset_type: str, req: Request):
    auth(req)
    if asset_type not in ("vivienda", "mototaxi", "maquinaria", "terreno", "vehiculo"):
        raise HTTPException(422, "Tipo de activo desconocido")
    return {"asset_type": asset_type, "shots": guided_shots(asset_type),
            "instructions": "Captura las vistas guiadas sin filtros, buena iluminacion y sin ocultar daños. La IA no puede verificar seguridad estructural ni aprobar creditos."}

@app.post("/api/cases/{case_id}/vision/{evidence_id}")
def vision_photo(case_id: str, evidence_id: str, req: Request):
    actor = auth(req, write=True)
    case = require_case(case_id)
    with db() as con:
        photo = con.execute("SELECT * FROM evidence WHERE id=? AND case_id=?", (evidence_id, case_id)).fetchone()
    if not photo:
        raise HTTPException(404, "La fotografia no pertenece al expediente")
    try:
        result = analyze_photo(str(UPLOADS / photo["filename"]), photo["shot"], case["asset_type"])
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        # Avoid leaking private inputs or provider credentials.
        raise HTTPException(503, str(exc)[:250]) from exc
    rid = str(uuid.uuid4())
    with db() as con:
        con.execute("INSERT INTO visual_assessments(id,case_id,evidence_id,state,result_json,created_at) VALUES(?,?,?,?,?,?)",
                    (rid,case_id,evidence_id,"pendiente",json.dumps(result,ensure_ascii=False),now()))
        con.execute("UPDATE cases SET updated_at=? WHERE id=?", (now(),case_id))
    log(actor,"analisis_visual_pendiente",case_id)
    return {"id":rid,"state":"pendiente","result":result,
            "warning":"Observaciones automaticas no verificadas; un revisor debe confirmarlas. Ningun credito se aprueba automaticamente."}

@app.post("/api/cases/{case_id}/automatic-inspection/{evidence_id}")
def automatic_inspection(case_id: str, evidence_id: str, req: Request):
    actor = auth(req, write=True)
    case = require_case(case_id)
    if case["asset_type"] != "vivienda":
        raise HTTPException(422, "El modelo disponible solo esta entrenado para fisuras de vivienda")
    if os.getenv("YOLO_CPU_ENABLED") != "1":
        raise HTTPException(503, "Modelo YOLO CPU no habilitado en el servidor")
    with db() as con:
        photo = con.execute("SELECT * FROM evidence WHERE id=? AND case_id=?",
                            (evidence_id,case_id)).fetchone()
    if not photo:
        raise HTTPException(404, "Fotografia no registrada en este expediente")
    try:
        detections, annotated = predict_cracks(str(UPLOADS / photo["filename"]))
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        # Do not expose provider internals to unauthenticated clients.
        raise HTTPException(503, "Inferencia YOLO no disponible en este servidor") from exc
    overlay_path = UPLOADS / (evidence_id + "-yolo.jpg")
    annotated.save(overlay_path,format="JPEG",quality=88)
    result = {
        "condition":"requiere_mejoras" if detections else "no_determinable",
        "summary": ("YOLO encontro " + str(len(detections)) +
                    " posible(s) fisura(s). Hallazgos sujetos a revision humana.")
                   if detections else
                   "YOLO no encontro fisuras sobre el umbral de confianza. Esto NO acredita que la vivienda este en buen estado.",
        "materials": [],
        "issues": [{
             "category":d["category"], "severity":"leve",
             "description":d["description"]+" Confianza "+str(round(d["confidence"]*100))+"%.",
             "improvement":d["improvement"], "box":d["box"], "confidence":d["confidence"],
             "has_mask":d.get("has_mask",False),
             "mask_coverage_percent_image":d.get("mask_coverage_percent_image")
        } for d in detections],
        "limitations":[
            "YOLOv8n-seg entrenado para fisuras visibles: no identifica techo de calamina, ladrillo, tarrajeo ni humedad.",
            "No determina estabilidad estructural ni daños ocultos. Ausencia de deteccion no implica ausencia de daños.",
            "Se requiere validacion por perito para decisiones de credito o garantia."
        ],
        "method":"yolo_cpu_onnx", "model":"YOLOv8n fisuras",
        "annotated_url":"/api/evidence/"+evidence_id+"/yolo-overlay",
    }
    analysis_id = str(uuid.uuid4())
    with db() as con:
        con.execute("INSERT INTO visual_assessments(id,case_id,evidence_id,state,result_json,created_at) VALUES(?,?,?,?,?,?)",
                    (analysis_id,case_id,evidence_id,"pendiente",
                     json.dumps(result,ensure_ascii=False),now()))
        con.execute("UPDATE cases SET updated_at=? WHERE id=?",(now(),case_id))
    log(actor,"detectar_fisuras_yolo",case_id)
    return {"id":analysis_id,"state":"pendiente","result":result,
            "warning":"La IA solo identifica posibles fisuras. Ninguna deteccion valida una tasacion ni aprueba creditos."}

@app.get("/api/evidence/{evidence_id}/yolo-overlay")
def yolo_overlay(evidence_id: str, req: Request):
    auth(req)
    with db() as con:
        row = con.execute("SELECT filename FROM evidence WHERE id=?", (evidence_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Fotografia no encontrada")
    path = UPLOADS / (evidence_id + "-yolo.jpg")
    if not path.exists():
        raise HTTPException(404, "Imagen anotada no disponible. Ejecuta el analisis de nuevo.")
    return FileResponse(path,media_type="image/jpeg",headers={"Cache-Control":"private, no-store"})


@app.post("/api/cases/{case_id}/manual-inspection/{evidence_id}")
def manual_inspection(case_id: str, evidence_id: str, data: ManualVisualObservation, req: Request):
    actor = auth(req, write=True, review=True)
    require_case(case_id)
    if data.condition not in ("aparentemente_bueno", "requiere_mejoras", "no_determinable"):
        raise HTTPException(422, "Condicion visual invalida")
    if data.category and data.severity not in ("leve", "moderada", "severa"):
        raise HTTPException(422, "Severidad invalida")
    with db() as con:
        photo = con.execute("SELECT 1 FROM evidence WHERE id=? AND case_id=?",
                            (evidence_id,case_id)).fetchone()
    if not photo:
        raise HTTPException(404, "Evidencia no encontrada")
    result = {
        "condition": data.condition,
        "summary": data.notes or "Observación documentada por el revisor",
        "materials": [{"name":data.material, "location":"Fotografía revisada"}] if data.material.strip() else [],
        "issues": [{"category":data.category,"severity":data.severity,
                    "description":data.description,"improvement":data.improvement}]
                  if data.category.strip() else [],
        "limitations": ["Evaluación manual de una fotografía. No determina seguridad estructural ni estado de elementos ocultos."],
        "method": "manual_review",
    }
    ident = str(uuid.uuid4())
    with db() as con:
        con.execute("INSERT INTO visual_assessments(id,case_id,evidence_id,state,result_json,reviewer,reviewed_at,created_at) VALUES(?,?,?,?,?,?,?,?)",
                    (ident,case_id,evidence_id,"confirmado",json.dumps(result,ensure_ascii=False),
                     actor,now(),now()))
        con.execute("UPDATE cases SET updated_at=? WHERE id=?", (now(),case_id))
    log(actor,"observacion_visual_manual",case_id)
    return {"id":ident,"state":"confirmado","result":result,
            "warning":"Observación manual, no detección por IA. El score es provisional si faltan vistas."}


@app.put("/api/visual-assessments/{assessment_id}/review")
def review_visual(assessment_id: str, data: VisualReview, req: Request):
    actor = auth(req, write=True, review=True)
    if data.state not in ("confirmado","descartado"):
        raise HTTPException(422, "Selecciona confirmado o descartado")
    with db() as con:
        assessment = con.execute("SELECT case_id FROM visual_assessments WHERE id=?", (assessment_id,)).fetchone()
        if not assessment:
            raise HTTPException(404, "Analisis no encontrado")
        con.execute("UPDATE visual_assessments SET state=?,reviewer=?,reviewed_at=? WHERE id=?",
                    (data.state,actor,now(),assessment_id))
    log(actor,"revisar_analisis_visual",assessment["case_id"])
    return {"state":data.state}

@app.post("/api/cases/{case_id}/credit-simulation")
def credit_simulation(case_id: str, data: CreditSimulation, req: Request):
    actor = auth(req, review=True)
    case = serialize_case(require_case(case_id))
    if not case["source"].strip():
        return {"status":"pendiente","suggested_amount":None,"reason":"Falta documentar la fuente de valor base."}
    outcome = simulate_credit(case["visual_condition"],case["base_value"],data.documented_budget)
    log(actor,"simular_microcredito",case_id)
    return {**outcome,"currency":"PEN","budget_source":data.budget_source,
            "warning":"Simulacion academica, no vinculante. Requiere verificacion fisica, tasacion, analisis de pago y aprobacion de la entidad."}

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
             "Valor base ingresado por usuario: " + ("S/ " + format(case["base_value"],",.2f") if case["base_value"] > 0 else "No determinado"),
             "Estimacion manual condicionada: " + ("S/ " + format(case["estimated_value"],",.2f") if case["base_value"] > 0 else "No determinada"),
             "Estado: " + case["valuation_status"],
             "Fuente: " + case["source"][:100], "",
             "HALLAZGOS"]
    for f in case["findings"]:
        lines.append((f["category"] + " | " + f["severity"] + " | " + f["state"] + " | Ajuste S/ " + str(f["adjustment"]))[:105])
    quick = case.get("quick_report")
    if quick:
        lines += ["", "EVALUACION RAPIDA AUTOMATICA (YOLO, NO PERITAJE)",
                  "Fotos analizadas: " + str(quick.get("images_analyzed",0)),
                  "Estado visual: " + str(quick.get("condition","No determinable")),
                  "Posibles hallazgos: " + str(len(quick.get("issues",[])))]
        for item in quick.get("improvements",[])[:30]:
            lines.append(("Mejora propuesta: " + str(item.get("action","")))[:105])
        repair = quick.get("automatic_repair_scenario") or build_automatic_quote(quick.get("issues",[]))
        if repair.get("illustrative_min_pen") is not None:
            lines.append("Reparaciones, ejemplo NO medido: S/ " +
                         str(repair["illustrative_min_pen"]) + " a S/ " +
                         str(repair["illustrative_max_pen"]))
        else:
            lines.append("Reparaciones: sin rango cuantificable con la evidencia.")
        reference = quick.get("official_construction_reference")
        if reference:
            lines += ["Referencia de edificacion por datos declarados: S/ "+
                      str(reference["construction_reference_pen"]),
                      "No incluye terreno ni depreciacion; NO es valor comercial."]
        else:
            example = quick.get("illustrative_construction_example") or reference_scenario(60,"sierra","D",illustrative=True)
            lines += ["EJEMPLO HIPOTETICO (no valor de esta casa): S/ "+
                      str(example["construction_reference_pen"]),
                      "Supuesto de ejemplo: Sierra, 60 m2, categoria D."]
        lines += ["Pendiente: area real, comparables, estado estructural, terreno y revision."]

    visual = case["visual_condition"]
    lines += ["", "INSPECCION VISUAL (NO ESTRUCTURAL)",
              "Cobertura fotografica: " + str(visual["coverage"]) + "%",
              "Tomas revisadas: " + str(visual["reviewed"]) + "/" + str(len(visual["required_shots"])),
              "Score visual referencial: " + (str(visual["condition_score"]) + "/100" if visual["condition_score"] is not None else "No disponible")]
    for item in case["visual_assessments"]:
        if item["state"] != "confirmado":
            continue
        data = json.loads(item["result_json"])
        lines.append(("Observacion: " + data.get("summary", ""))[:105])
        for issue in data.get("issues", []):
            lines.append(("Mejora sugerida: " + issue.get("improvement", ""))[:105])
    lines += ["", "Sin verificacion de estabilidad, danos ocultos ni habitabilidad.",
              "No es aprobacion bancaria, score crediticio validado o peritaje."]
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
