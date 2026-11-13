"""Fast, evidence-based inspection summary; do not manufacture property values or area."""
import json
import os
from inspection_taxonomy import CLASSES, public_capabilities

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
                    "confidence":issue.get("confidence"),
                    "shot":a["shot"],
                    "label":CLASSES.get(str(issue.get("category","")),{ }).get("label",str(issue.get("category",""))),
                    "urgency":CLASSES.get(str(issue.get("category","")),{ }).get("urgency","revisar")
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
    multiclass = any("defect5" in model for model in models)
    detail = ("YOLO multiclase reconoce cinco categorías de defectos observables, pero su "
              "precisión es limitada en humedad, moho y pintura. No identifica puerta ausente, "
              "tarrajeo, pisos fracturados, calamina o materiales sin pesos adicionales."
              if multiclass else
              "El detector actual solo está entrenado para posibles fisuras; "
              "otros daños y materiales no se pueden verificar con este modelo.")
    improvements=[]
    for issue in issues:
        ident=issue["category"]+"@"+issue["shot"]
        if not any(item["id"]==ident for item in improvements):
            improvements.append({
                "id":ident, "shot":issue["shot"],"category":issue["category"],
                "label":issue["label"],"priority":issue["urgency"],
                "action":issue["improvement"] or CLASSES.get(issue["category"],{}).get("improvement","Realizar inspección profesional"),
                "before_quoting":"Confirmar el diagnóstico, tomar metrados y solicitar cotización local de materiales y mano de obra.",
                "confirmed":False,
            })
    alert_priority=("prioritaria" if any(x["urgency"]=="prioritaria" for x in issues)
                    else "revisar" if issues else "no_determinable")
    return {
        "status": "preliminar_automatico",
        "images_analyzed":photo_count,
        "condition":condition,
        "materials":materials,
        "issues":issues,
        "improvements":improvements,
        "inspection_priority":alert_priority,
        "capabilities":public_capabilities(multiclass),
        "image_summaries":notes,
        "models":sorted(set(models)),
        "analysis_errors":errors,
        "limitations":LIMITATIONS+[detail],
        "quote": build_quote(issues),
        "property_market_value":None,
        "property_market_value_reason":"Una tasación monetaria requiere área construida/terreno, ubicación, materiales comprobados, estado contrastado, comparables o valores oficiales, y revisión profesional. Dos fotos no proporcionan esos datos.",
        "valuation_inputs_needed":["Ubicación exacta del inmueble","Área construida y del terreno","Tipo y antigüedad de construcción","Comparables o valores oficiales por región y categoría","Metrados y cotizaciones para mejoras"],
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
