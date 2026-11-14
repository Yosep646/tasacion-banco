"""Automatic, source-backed *illustrative* repair scenarios for visual evidence.

NO real-world surface area, length, diagnosis, labour minimum or material condition
can be inferred from a photo without calibration and inspection.
These are one-to-three-unit EXAMPLES, not a binding quote or credit approval.
"""
from decimal import Decimal, ROUND_HALF_UP
import math

# Stable, dated public online reference prices (PEN); hyperlinks are displayed
# alongside each line item. Conditions of the published assemblies matter.
CATALOG = {
    "fisura_visible": {
        "Exterior": {
            "code": "RYY022", "work": "Sellado de fisura EN REVESTIMIENTO de mortero exterior",
            "unit": "m", "unit_price_pen": "23.02",
            "source": "CYPE Perú, RYY022, sellado elástico de revestimiento vertical exterior",
            "source_url": "https://peru.generadordeprecios.info/rehabilitacion/Revestimientos/RY_Preparaciones__protecciones__r/Reparaciones/RYY022_Reparacion_de_grieta_en_revestimien.html",
            "note": "Sólo aplica si un profesional confirma fisura superficial del revestimiento; NO cubre reparación estructural ni repintado.",
        },
        "Interior": {
            "code": "RYY022", "work": "Sellado de fisura EN REVESTIMIENTO de mortero interior",
            "unit": "m", "unit_price_pen": "22.29",
            "source": "CYPE Perú, RYY022, sellado elástico de revestimiento vertical interior",
            "source_url": "https://peru.generadordeprecios.info/rehabilitacion/Revestimientos/RY_Preparaciones__protecciones__r/Reparaciones/RYY022_Reparacion_de_grieta_en_revestimien_0_0_0_0_0_0_0_0_0.html",
            "note": "Sólo aplica si se verifica una fisura no estructural en revestimiento de mortero.",
        },
    },
    "humedad_visible": {
        "default": {
            "code": "NHM010", "work": "Tratamiento de humedad por CAPILARIDAD con mortero (hipótesis)",
            "unit": "m²", "unit_price_pen": "102.88",
            "source": "CYPE Perú, NHM010, mortero de reparación para humedad por capilaridad",
            "source_url": "https://peru.generadordeprecios.info/rehabilitacion/Aislamientos_e_impermeabilizaciones/NH_Tratamiento_de_humedades_por_c/Morteros/NHM010_Tratamiento_de_humedades_por_capila.html",
            "note": "Una mancha NO prueba humedad capilar. Sólo ejemplo condicional; no incluye eliminar el revestimiento anterior, acabado posterior ni corregir la causa de entrada de agua.",
        },
    },
    "moho_visible": {
        "default": {
            "code": "PE-MOHO", "work": "Limpieza y tratamiento superficial de moho (mano de obra)",
            "unit": "m²", "unit_price_pen": "44.00",
            "source": "Vistobra Perú, tabla de pintores noviembre 2026, tratamiento de moho",
            "source_url": "https://vistobra.com/pe/precios/pintor",
            "note": "Mano de obra referencial: NO incluye diagnóstico, corregir infiltración ni materiales adicionales si corresponden.",
        },
    },
    "pintura_desprendida": {
        "default": {
            "code": "PE-PINTURA", "work": "Retiro de pintura suelta y repintado estándar de un paño",
            "unit": "m²", "unit_price_pen": "74.00",
            "source": "Vistobra Perú, raspado S/13/m² (mano de obra) + pintura estándar con material S/61/m², noviembre 2026",
            "source_url": "https://vistobra.com/pe/precios/pintor",
            "note": "Ejemplo de dos trabajos combinados. No cubre reparación del soporte, humedad, andamios ni transporte.",
        },
    },
}
EXCLUDED = {
    "grieta_escalonada": "Posible grieta estructural/escalonada: evaluar con ingeniero antes de proponer un precio; no aplicar sellado cosmético.",
    "pintura_deteriorada": "Detector de pintura experimental: excluido hasta validar su precisión en viviendas reales.",
}
MAX_PROVISIONAL_CONFIDENCE_FLOOR=0.5


def _money(value):
    return float(Decimal(value).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP))


def _safe_confidence(value):
    if isinstance(value,bool) or value is None:return None
    try:
        confidence=float(value)
    except (TypeError,ValueError):
        return None
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return None
    return confidence


def build_automatic_quote(issues):
    """Build an explicit 1–3 unit *scenario* using deduplicated YOLO issues.

    Do not multiply image boxes by real dimensions or assume damage area.
    No quantitative quote is returned when detections are unreliable.
    """
    lines=[]
    excluded=[]
    seen=set()
    rejected_low=0
    for issue in issues:
        if not isinstance(issue,dict):continue
        category=str(issue.get("category","")).lower().strip()
        shot=str(issue.get("shot","")).strip().capitalize()
        key=(category,shot)
        if key in seen:continue
        seen.add(key)
        confidence=_safe_confidence(issue.get("confidence"))
        if category in EXCLUDED:
            excluded.append({"category":category,"shot":shot,"reason":EXCLUDED[category]})
            continue
        if category not in CATALOG:
            excluded.append({"category":category,"shot":shot,"reason":"Sin partida comparable verificada para cotización automática."})
            continue
        if confidence is None or confidence<MAX_PROVISIONAL_CONFIDENCE_FLOOR:
            rejected_low+=1
            excluded.append({"category":category,"shot":shot,"reason":"Detección sin confianza verificable o inferior al 50 %; no se incluye en el rango."})
            continue
        variants=CATALOG[category]
        spec=variants.get(shot,variants.get("default"))
        if spec is None:
            excluded.append({"category":category,"shot":shot,"reason":"No hay partida para esta vista."})
            continue
        unit_price=Decimal(spec["unit_price_pen"])
        lines.append({
            "category":category,"shot":shot,"confidence":round(confidence,4),
            "work":spec["work"],"reference_code":spec["code"],
            "unit":spec["unit"],"unit_price_pen":_money(unit_price),
            "source":spec["source"],"source_url":spec["source_url"],"note":spec["note"],
            "quantity":None,"subtotal_pen":None,
            "illustrative_quantity_min":1,"illustrative_quantity_max":3,
            "illustrative_min_pen":_money(unit_price),
            "illustrative_max_pen":_money(unit_price*3),
            "requires_measurement":True,"confirmed":False,
        })
    low=sum((Decimal(str(item["illustrative_min_pen"])) for item in lines),Decimal("0"))
    high=sum((Decimal(str(item["illustrative_max_pen"])) for item in lines),Decimal("0"))
    return {
        "currency":"PEN",
        "status":"escenario_automatico_ilustrativo" if lines else (
            "sin_partidas_cuantificables" if issues else "sin_danos_diagnosticados"),
        "items":lines,"excluded":excluded,"low_confidence_excluded":rejected_low,
        "illustrative_min_pen":_money(low) if lines else None,
        "illustrative_max_pen":_money(high) if lines else None,
        "example_quantity_per_partida":"1 a 3 m o m² por partida, SUPUESTO DIDÁCTICO",
        "range_is_illustrative":True,
        "total_pen":None,
        "suggested_credit_pen":None,
        "source_policy":"Referencias de precios de Perú publicadas por CYPE y Vistobra; comprobar tarifa y ámbito antes de contratar.",
        "explanation":(
            "RANGO ILUSTRATIVO AUTOMÁTICO de partidas visibles con confianza >=50 %. "
            "Se aplica 1 a 3 m o m² por tipo de reparación y vista, SIN medir la foto, "
            "agrupar cajas de la misma categoría ni certificar la patología. "
            "No incluye todos los trabajos, gastos mínimos de visita, impuestos adicionales, "
            "costos de corregir causas ni intervención estructural. "
            "El costo final exige metrados y presupuesto profesional; no es una tasación."
        ),
    }
