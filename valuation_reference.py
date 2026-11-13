"""Auditable non-binding construction / land scenario. NOT an appraisal from photographs."""
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

def _money(value):
    return float(Decimal(value).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP))

def estimate_property_scenario(area_m2, construction_pen_m2, source,
                               land_area_m2=0, land_pen_m2=0, land_source=""):
    try:
        area=Decimal(str(area_m2)); construction=Decimal(str(construction_pen_m2))
        land_area=Decimal(str(land_area_m2)); land_unit=Decimal(str(land_pen_m2))
    except (InvalidOperation,ValueError,TypeError) as exc:
        raise ValueError("Valores numéricos inválidos") from exc
    if not all(x.is_finite() for x in [area,construction,land_area,land_unit]):
        raise ValueError("No se admiten valores infinitos")
    if not Decimal("1")<=area<=Decimal("100000"):
        raise ValueError("Área construida fuera de rango")
    if not Decimal("1")<=construction<=Decimal("100000"):
        raise ValueError("Precio documentado por m² fuera de rango")
    if not Decimal("0")<=land_area<=Decimal("100000"):
        raise ValueError("Área de terreno fuera de rango")
    if not Decimal("0")<=land_unit<=Decimal("1000000"):
        raise ValueError("Valor de terreno por m² fuera de rango")
    if not isinstance(source,str) or len(source.strip())<8:
        raise ValueError("Documenta la fuente y fecha del costo unitario")
    if (land_area>0 or land_unit>0) and (land_area<=0 or land_unit<=0 or len(land_source.strip())<8):
        raise ValueError("Para incluir terreno registra área, valor unitario y fuente")
    base=area*construction
    ground=land_area*land_unit
    return {
        "status":"escenario_documentado_no_pericial",
        "currency":"PEN",
        "built_area_m2":float(area),"construction_unit_pen_m2":float(construction),
        "construction_value_pen":_money(base),
        "construction_source":source.strip(),
        "land_area_m2":float(land_area),"land_unit_pen_m2":float(land_unit),
        "land_value_pen":_money(ground),"land_source":land_source.strip(),
        "indicative_total_pen":_money(base+ground),
        "disclaimer":"Escenario aritmético con precios aportados y fuente documentada. No constituye valor comercial certificado, avalúo pericial, clasificación física automática ni elegibilidad crediticia.",
        "formula":"área construida × valor unitario documentado + área de terreno × valor unitario documentado",
    }
