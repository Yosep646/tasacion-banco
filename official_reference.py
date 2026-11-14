"""Illustrative construction-only calculation from CAP 2026 November official unit tables.

A single category applied to all 7 components is a SIMPLIFIED, user-selected
arithmetic scenario, NOT an appraisal, a property price, market comparables or
evidence that the images depict those construction materials.
No estimate of land, structural condition, depreciation or credit suitability.
"""
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

PARTS=("muros_columnas","techos","pisos","puertas_ventanas","revestimientos","banos","instalaciones")
SOURCE_URLS={
    "lima_callao":"https://cap.org.pe/wp-content/uploads/2026/11/CVU_NOVIEMBRE_LIMA_METROPOLITANA_2026.pdf",
    "costa":"https://cap.org.pe/wp-content/uploads/2026/11/CVU_NOVIEMBRE_COSTA_2026.pdf",
    "sierra":"https://cap.org.pe/wp-content/uploads/2026/11/CVU_NOVIEMBRE_SIERRA_2026.pdf",
    "selva":"https://cap.org.pe/wp-content/uploads/2026/11/CVU_NOVIEMBRE_SELVA_2026.pdf",
}
# Each 7-tuple is transcribed in the exact column order of the CAP tables.
VALUES={
 "lima_callao":{
  "C":("305.04","225.01","145.44","127.08","225.97","71.56","183.66"),
  "D":("294.99","149.01","128.29","111.31","173.38","38.18","116.02"),
  "E":("207.67","53.25","85.96","95.24","119.29","22.44","84.26"),
  "F":("156.40","29.28","58.70","71.50","84.09","16.72","48.20"),
 },
 "costa":{
  "C":("301.19","222.17","143.59","125.47","223.12","70.66","174.44"),
  "D":("291.26","147.13","126.67","109.90","171.19","37.70","110.42"),
  "E":("205.05","52.57","84.87","94.04","117.78","22.16","80.26"),
  "F":("154.42","28.91","57.96","70.59","83.02","16.51","44.14"),
 },
 "sierra":{
  "C":("326.31","189.11","150.50","192.64","248.87","62.28","208.25"),
  "D":("301.39","143.11","123.40","112.99","190.37","38.11","118.00"),
  "E":("236.61","58.77","102.06","86.31","158.37","18.68","65.67"),
  "F":("147.55","46.95","83.35","66.75","94.43","15.88","42.69"),
 },
 "selva":{
  "C":("399.06","216.12","155.46","202.71","233.86","72.77","214.26"),
  "D":("308.55","188.43","131.80","135.88","169.00","49.34","119.15"),
  "E":("244.99","137.19","106.34","88.23","128.14","24.49","80.56"),
  "F":("193.20","63.08","86.59","71.96","99.07","20.82","44.49"),
 },
}

def _pen(value):
    return float(value.quantize(Decimal("0.01"),rounding=ROUND_HALF_UP))

def official_unit_values():
    return {"year":2026,"month":"noviembre","country":"Perú","currency":"PEN",
            "status":"tabla_referencial_no_comercial",
            "source_urls":SOURCE_URLS,
            "zones":{zone:{grade:_pen(sum(map(Decimal,row))) for grade,row in categories.items()}
                     for zone,categories in VALUES.items()},
            "method":"Suma de las 7 partidas usando la misma categoría elegida solo como SIMULACIÓN.",
            "required_for_real_property":["área techada medida","zona real del inmueble",
                                          "categorías comprobadas de las 7 partidas",
                                          "antigüedad y depreciación","valor del terreno y comparables"]}

def reference_scenario(area_m2, zone, grade, *, illustrative=False):
    if zone not in VALUES or grade not in VALUES[zone]:
        raise ValueError("Elige una zona y categoría permitidas de la tabla oficial")
    try:
        area=Decimal(str(area_m2))
    except (ValueError,TypeError,InvalidOperation) as exc:
        raise ValueError("Área construida inválida") from exc
    if not area.is_finite() or not Decimal("1")<=area<=Decimal("50000"):
        raise ValueError("Área techada debe estar entre 1 y 50 000 m²")
    vals=[Decimal(x) for x in VALUES[zone][grade]]
    unit=sum(vals)
    subtotal=area*unit
    return {
       "status":"simulacion_ejemplo_no_propiedad" if illustrative else "referencia_oficial_simplificada",
       "source":"CAP, Cuadro de valores unitarios oficiales de edificación, noviembre 2026",
       "source_url":SOURCE_URLS[zone],"currency":"PEN","zone":zone,"grade":grade,
       "year":2026,"month":"noviembre","built_area_m2":float(area),
       "unit_cost_pen_m2":_pen(unit),"construction_reference_pen":_pen(subtotal),
       "partitions":{k:_pen(v) for k,v in zip(PARTS,vals)},
       "land_value_pen":None,"market_value_pen":None,"suggested_credit_pen":None,
       "area_user_reported":not illustrative,
       "grade_verified":False,
       "method":"Área techada × suma de siete partidas de una categoría uniforme elegida para simular.",
       "disclaimer":(
           "SIMULACIÓN DE EJEMPLO, no calcula el valor de esta casa." if illustrative else
           "Referencia aritmética basada en área declarada y categoría homogénea no verificada."
       )+" No es precio comercial, tasación hipotecaria, valor del terreno, depreciación, costo de reparación ni monto de crédito. La categoría de cada partida requiere evaluación técnica.",
    }
