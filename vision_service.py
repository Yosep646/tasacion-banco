"""Visual inspection adapter for property and collateral photographs.

This is a *visual* observation, not a structural or banking appraisal.
External vision is opt-in via OPENAI_API_KEY. No invented image inference.
"""
import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

GUIDES = {
    "vivienda": ["Fachada", "Paredes", "Techo", "Interior"],
    "mototaxi": ["Frente", "Laterales", "Motor", "Interior"],
    "vehiculo": ["Frente", "Laterales", "Interior", "Motor"],
    "maquinaria": ["General", "Componentes", "Desgaste", "Placa"],
    "terreno": ["Acceso", "Linderos", "Superficie", "Entorno"],
}
SEVERITIES = {"leve", "moderada", "severa"}
CONDITIONS = {"aparentemente_bueno", "requiere_mejoras", "no_determinable"}
SYSTEM_PROMPT = """Eres asistente de inspeccion visual para microcredito en Peru.
Examina SOLO la imagen proporcionada, sin inventar caracteristicas.
Identifica materiales visibles (ej. calamina, concreto, ladrillo expuesto,
pared sin tarrajear) y problemas visibles (fisuras, humedad superficial,
oxidacion, abolladuras, desgaste) cuando haya evidencia observable.
Nunca declares seguridad estructural, estado electrico interno,
habitabilidad, valor economico ni elegibilidad bancaria a partir de fotos.
No infieras causas, severidades estructurales ni daños ocultos.
Si la imagen no permite verificar algo, usa no_determinable o elementos vacios.
No confundas sombras con fisuras. Responde SOLO JSON con las claves:
condition: aparentemente_bueno|requiere_mejoras|no_determinable
summary: texto en espanol (max 300 caracteres)
materials: lista de {name, location} (max 8)
issues: lista de {category,severity,description,improvement} (max 8),
severity: leve|moderada|severa; severidad visual preliminar.
limitations: lista de textos breves (max 5)
La recomendacion debe ser inspeccion o mejora, no monto ni autorizacion de credito.
"""

def guided_shots(asset_type):
    return GUIDES.get(asset_type, GUIDES["vivienda"])


def normalize_vision(payload):
    if not isinstance(payload, dict):
        raise ValueError("Respuesta visual invalida")
    condition = str(payload.get("condition", "no_determinable")).strip()
    if condition not in CONDITIONS:
        condition = "no_determinable"
    materials = []
    for item in payload.get("materials", [])[:8]:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()[:100]
            if name:
                materials.append({"name": name, "location": str(item.get("location", ""))[:150]})
    issues = []
    for item in payload.get("issues", [])[:8]:
        if isinstance(item, dict):
            category = str(item.get("category", "")).strip()[:120]
            if not category:
                continue
            severity = str(item.get("severity", "moderada")).lower()
            issues.append({
                "category": category,
                "severity": severity if severity in SEVERITIES else "moderada",
                "description": str(item.get("description", ""))[:500],
                "improvement": str(item.get("improvement", ""))[:300],
            })
    return {"condition": condition, "summary": str(payload.get("summary", ""))[:350],
            "materials": materials, "issues": issues,
            "limitations": [str(x)[:200] for x in payload.get("limitations", [])[:5] if isinstance(x, str)]}


def extract_response_text(result):
    parts = []
    for item in result.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    value = "".join(parts).strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if not value:
        raise ValueError("El modelo no devolvio observaciones")
    return value


def analyze_photo(image_path, shot, asset_type):
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Vision multimodal no configurada: falta OPENAI_API_KEY en Render. No se simularan resultados.")
    raw = Path(image_path).read_bytes()
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("Imagen excede el limite permitido")
    model = os.getenv("VISION_MODEL", "gpt-4.1-mini").strip()
    if not model:
        raise RuntimeError("VISION_MODEL vacio")
    prompt = ("Activo: " + asset_type + ". Toma: " + shot
              + ". Describe solo lo que se ve en esta foto; marca incertidumbres.")
    body = {"model": model, "instructions": SYSTEM_PROMPT,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": "data:image/jpeg;base64," +
                 base64.b64encode(raw).decode("ascii"), "detail": "low"}
            ]}], "max_output_tokens": 1200,
            "text": {"format": {"type": "json_object"}}}
    req = urllib.request.Request("https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=50) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        # Do not log credentials, image content or personal data from error bodies.
        raise RuntimeError("Proveedor visual respondio HTTP " + str(exc.code)) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("No se pudo conectar al proveedor de vision") from exc
    return normalize_vision(json.loads(extract_response_text(result)))


def evaluate_condition(asset_type, evidence, assessments, findings):
    """Transparent rule-based condition score, only for reviewed evidence.

    Coverage is four guided views. Reviewers must approve all latest views.
    The result is NOT a diagnosis or a lender's underwriting score.
    """
    shots = guided_shots(asset_type)
    by_shot = {}
    for item in evidence:
        name = item["shot"].strip().casefold()
        if name in (x.casefold() for x in shots) and name not in by_shot:
            by_shot[name] = item["id"]
    latest = {a["evidence_id"]: a for a in reversed(assessments)}
    reviewed = [latest[eid] for eid in by_shot.values()
                if eid in latest and latest[eid]["state"] == "confirmado"]
    pending = [latest[eid] for eid in by_shot.values()
               if eid in latest and latest[eid]["state"] == "pendiente"]
    issues = [issue for assessment in reviewed
              for issue in json.loads(assessment["result_json"]).get("issues", [])]
    deductions = sum({"leve": 5, "moderada": 12, "severa": 25}.get(x["severity"], 12)
                     for x in issues)
    known = sum(json.loads(x["result_json"]).get("condition") != "no_determinable"
                for x in reviewed)
    # One reviewed photograph is enough for a provisional visual assessment,
    # but never for a validated loan suggestion.
    provisional = bool(reviewed) and known == len(reviewed)
    complete = (len(by_shot) == len(shots)
                and len(reviewed) == len(shots)
                and known == len(shots))
    return {
        "required_shots": shots, "captured": len(by_shot), "reviewed": len(reviewed),
        "pending": len(pending), "coverage": round(100 * len(by_shot) / len(shots)),
        "condition_score": max(0, 100 - deductions) if provisional else None,
        "score_status": ("referencial_revisado" if complete
                         else "provisional_parcial" if provisional else "insuficiente"),
        "credit_ready": complete,
        "issue_count": len(issues),
        "explanation": "Puntaje visual ilustrativo sobre fotos revisadas: 100; -5 leve, -12 moderada, -25 severa. Con menos de cuatro vistas es parcial; no es score crediticio.",
    }


def simulate_credit(condition, base_value, documented_budget, fraction=0.7):
    """Illustrative improvement loan range, never an approval."""
    if not condition.get("credit_ready", False) or base_value <= 0 or documented_budget <= 0:
        return {"status": "pendiente", "suggested_amount": None,
                "reason": "La simulación requiere las cuatro vistas revisadas, valor base con fuente y presupuesto documentado; una sola foto produce evaluación parcial."}
    amount = min(documented_budget * fraction, base_value * 0.3)
    return {"status": "simulacion_no_vinculante", "suggested_amount": round(amount, 2),
            "reason": "Ejemplo configurable: 70% del presupuesto o 30% del valor base, el menor. No es politica de una entidad financiera."}
