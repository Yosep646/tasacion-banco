"""Taxonomía documentada para inspección visual de viviendas.

Los modelos solo pueden emitir clases para las que existen pesos compatibles.
La ausencia de una detección NUNCA prueba que un elemento exista o esté sano.
"""
CLASSES = {
    "fisura_visible": {"label":"Posible fisura superficial", "group":"estructura_visible",
        "improvement":"Verificar ancho y evolución de la fisura; solicitar reparación técnica según diagnóstico.",
        "urgency":"revisar", "unit":"metro lineal"},
    "grieta_escalonada": {"label":"Posible grieta escalonada", "group":"estructura_visible",
        "improvement":"Solicitar evaluación presencial prioritaria por especialista estructural antes de intervenir.",
        "urgency":"prioritaria", "unit":"metro lineal"},
    "humedad_visible": {"label":"Posible filtración o mancha de humedad", "group":"acabados",
        "improvement":"Inspeccionar origen de la filtración, corregirlo, secar y rehabilitar acabado afectado.",
        "urgency":"revisar", "unit":"metro cuadrado"},
    "moho_visible": {"label":"Posible moho superficial", "group":"acabados",
        "improvement":"Verificar humedad y ventilación; efectuar saneamiento seguro antes de repintar.",
        "urgency":"revisar", "unit":"metro cuadrado"},
    "pintura_desprendida": {"label":"Pintura o revestimiento desprendido", "group":"acabados",
        "improvement":"Retirar material suelto, preparar la superficie, aplicar sellador y repintar según evaluación.",
        "urgency":"normal", "unit":"metro cuadrado"},
    "pintura_deteriorada": {"label":"Pintura deteriorada", "group":"acabados",
        "improvement":"Preparar superficie y repintar después de verificar humedad.", "urgency":"normal", "unit":"metro cuadrado"},
    "piso_roto": {"label":"Piso o baldosa fracturados", "group":"pisos",
        "improvement":"Inspeccionar soporte y sustituir piezas dañadas.", "urgency":"revisar", "unit":"metro cuadrado"},
    "pared_sin_tarrajeo": {"label":"Pared sin tarrajear", "group":"acabados",
        "improvement":"Verificar soporte y presupuestar tarrajeo y acabado.", "urgency":"normal", "unit":"metro cuadrado"},
    "pared_tarrajeada": {"label":"Pared tarrajeada visible", "group":"material",
        "improvement":"Sin reparación deducible solo por presencia de tarrajeo.", "urgency":"informativo","unit":None},
    "ladrillo_expuesto": {"label":"Ladrillo visible", "group":"material",
        "improvement":"Determinar si requiere protección o tarrajeo.", "urgency":"informativo","unit":None},
    "puerta_visible": {"label":"Puerta observada", "group":"elemento",
        "improvement":"Verificar funcionamiento y estado en inspección.", "urgency":"informativo","unit":None},
    "hueco_sin_puerta": {"label":"Posible vano sin puerta", "group":"elemento",
        "improvement":"Confirmar si corresponde instalar una puerta en ese vano.", "urgency":"normal","unit":"unidad"},
    "calamina_visible": {"label":"Cobertura de calamina", "group":"material",
        "improvement":"Verificar corrosión, anclajes y filtraciones.", "urgency":"informativo","unit":None},
    "techo_concreto_visible": {"label":"Techo aparente de concreto", "group":"material",
        "improvement":"Confirmar tipo de cobertura en inspección.", "urgency":"informativo","unit":None},
}
# Source labels from the actual pretrained YOLO26n object detector.
PRETRAINED_5 = {
    "crack":"fisura_visible", "mold":"moho_visible",
    "peeling_paint":"pintura_desprendida",
    "stairstep_crack":"grieta_escalonada", "water_seepage":"humedad_visible",
}
# No weights are provided or claimed for these classes yet.
PENDING_TRAINING = (
    "piso_roto","pared_sin_tarrajeo","pared_tarrajeada","ladrillo_expuesto",
    "puerta_visible","hueco_sin_puerta","calamina_visible",
    "techo_concreto_visible","pintura_deteriorada",
)
def improvement_for(category):
    return CLASSES.get(category,{}).get("improvement","Evaluar y documentar reparación con especialista.")

def public_capabilities(multiclass_enabled):
    return {
        "automatic": sorted(set(PRETRAINED_5.values()) | {"fisura_visible"}) if multiclass_enabled
                      else ["fisura_visible"],
        "training_required": list(PENDING_TRAINING),
        "note": "Las categorías pendientes NO se infieren con el detector actual. No detectar una puerta no prueba su ausencia.",
    }
