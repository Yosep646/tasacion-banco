"""Fast, evidence-based inspection summary; do not manufacture property values or area."""
import json
import os

LIMITATIONS = [
    "Una o dos fotos no muestran el estado completo de la vivienda.",
    "No se verifican cimentación, instalaciones ocultas, estabilidad, humedad interna ni documentación.",
    "Una detección visual no implica diagnóstico estructural ni aprobación de crédito.",
]

def summarize(assessments, photo_count, provider_available):
    materials = []
    issues = []
    notes = []
    models = []
    errors = []
    visible_conditions = []
    for a in assessments:
        result = a["result"]
        models.append(result.get("method", "desconocido"))
        visible_conditions.append(result.get("condition", "no_determinable"))
        for material in result.get("materials", []):
            if isinstance(material, dict):
                name = str(material.get("name", "")).strip()
                if name and name.casefold() not in [v["name"].casefold() for v in materials]:
                    materials.append({"name":name,"location":str(material.get("location",""))})
        for issue in result.get("issues", []):
            if isinstance(issue, dict):
                issues.append({
                    "category":str(issue.get("category", "Daño visible"))[:120],
                    "severity":str(issue.get("severity", "no_determinada"))[:30],
                    "description":str(issue.get("description",""))[:500],
                    "improvement":str(issue.get("improvement",""))[:300],
                    "evidence_id":a["evidence_id"],
                    "confidence":issue.get("confidence")
                })
        notes.append({"shot":a["shot"],"summary":result.get("summary",""),
                      "evidence_id":a["evidence_id"],
                      "annotated_url":result.get("annotated_url")})
        if a.get("error"):
            errors.append(a["error"])
    if issues:
        condition = "Posibles mejoras o inspecciones necesarias"
    elif any(c=="aparentemente_bueno" for c in visible_conditions):
        condition = "Sin problemas visibles identificados en las vistas analizadas; revisión incompleta"
    else:
        condition = "No determinable con las imágenes disponibles"
    detail = ("El detector YOLO disponible está entrenado exclusivamente para posibles fisuras. "
              "Otros materiales y deterioros requieren un modelo multimodal configurado; "
              "su ausencia no puede interpretarse como que no existen defectos.")
    return {
        "status": "preliminar_automatico",
        "images_analyzed":photo_count,
        "condition":condition,
        "materials":materials,
        "issues":issues,
        "image_summaries":notes,
        "models":sorted(set(models)),
        "analysis_errors":errors,
        "limitations":LIMITATIONS+[detail],
        "quote": build_quote(issues),
        "property_market_value":None,
        "property_market_value_reason":"No es posible estimar un valor comercial preciso con 1-2 imágenes: faltan ubicación verificada, área, documentos y comparables.",
        "human_review_required":True,
    }

def build_quote(issues):
    """Quote individual interventions ONLY when a documented approved price catalog exists.
    Unknown quantities cannot be treated as 1m², and no whole-property value is invented.
    """
    try:
        catalog = json.loads(os.environ.get("TASACION_PRICE_CATALOG", "{}"))
        if not isinstance(catalog, dict):
            catalog = {}
    except (ValueError, TypeError):
        catalog = {}
    lines = []
    categories = {str(x.get("category","")).casefold() for x in issues}
    for category, item in catalog.items():
        if category.casefold() not in categories or not isinstance(item,dict):
            continue
        try:
            unit_price = float(item["unit_price_pen"])
        except (ValueError, TypeError, KeyError):
            continue
        if not 0 < unit_price < 1000000:
            continue
        source = str(item.get("source","")).strip()
        unit = str(item.get("unit","")).strip()
        if not source or not unit:
            continue
        lines.append({"work":category, "unit":unit, "unit_price_pen":round(unit_price,2),
                      "source":source,"quantity":None,"subtotal_pen":None})
    return {
        "currency":"PEN",
        "status":"requiere_metrado_y_cotizacion" if issues else "sin_trabajos_cuantificables",
        "items":lines,
        "total_pen":None,
        "explanation":"Se muestra precio unitario únicamente si hay un catálogo con fuente documentada. El total requiere metrados, alcance y verificación; no se calcula desde fotos.",
        "suggested_credit_pen":None,
    }
