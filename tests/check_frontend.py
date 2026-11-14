"""Basic HTML/JS checks for the single-file dashboard, no browser required."""
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import tempfile

class ScriptCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_script = False
        self.scripts = []
    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.in_script = True
    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False
    def handle_data(self, data):
        if self.in_script:
            self.scripts.append(data)

p=Path(__file__).resolve().parents[1] / "static" / "index.html"
html=p.read_text(encoding="utf-8")
parser=ScriptCollector()
parser.feed(html)
script="\n".join(parser.scripts)
assert script.strip(), "No se encontró JavaScript en el dashboard"
for required in ('id="caseForm"', 'id="caseList"', 'id="caseSearch"', 'name="asset_type"', 'name="title"', 'name="location"'):
    assert required in html or required.replace('"', '\\"') in html, required
# Integration of the chosen first design must preserve functional quick evaluation.
for required in (
    'class="one-hero"',
    'class="one-main-grid"',
    'id="quickForm"',
    'id="quickAnalyzeBtn"',
    'id="exteriorFile"',
    'id="interiorFile"',
    'id="caseList"',
    'id="logout"',
    '/api/quick-inspection',
    'id="officialReferenceForm"',
    'id="referencePreview"',
    '/official-construction-reference',
    'id="openRapidReport"',
    'renderQuickResults(body)',
    'multiclassYoloEnabled?"YOLO 5 defectos configurado"',
    '/api/model-readiness',
):
    assert required in html, "Falta elemento del diseño integrado: " + required
# One compact summary replaces duplicated findings/improvements/material cards.
for required in (
    'class="inspection-unified"',
    'class="compact-finding"',
    'class="compact-technical"',
    'grouped=new Map()',
    'id="repairRange"',
    'id="referenceTotal"',
    'id="takeExterior"',
    'id="takeInterior"',
    'id="uploadExterior"',
    'id="uploadInterior"',
    'id="nativeExterior"',
    'id="nativeInterior"',
    'openPhotoCamera(',
    "navigator.mediaDevices?.getUserMedia",
    'capture="environment"',
    "photos.append('exterior',selectedPhotos.exterior",
):
    assert required in html, "Falta módulo compacto o captura de foto: " + required
assert 'class="result-two-col"' not in html[html.index('function renderQuickResults(result){'):]
assert 'class="result-improvements"' not in html[html.index('function renderQuickResults(result){'):]
assert "stream.getTracks().forEach(track=>track.stop())" in html
for required in (
    'Simulación automática de edificación (EJEMPLO)',
    'constructionExplainer',
    'id="referenceCardExplanation"',
    'class="construction-source-link"',
    'class="repair-quote-details"',
    'id="referenceAdjust"',
    'official-reference-adjust',
):
    assert required in html, "Falta bloque de simulación automática: " + required
assert "'No calculado para esta vivienda'" not in html
# The advanced case tabs now consume the stored quick YOLO report directly.
for required in (
    'current.quick_findings_need_sync',
    '/sync-quick-report',
    'const q=c.quick_report||null',
    'Reparaciones · escenario ilustrativo',
    'Hallazgos registrados automáticamente',
    'Gravedad no determinada',
    'class="case-manual-details"',
    'Resultado económico de la evaluación rápida',
    'PDF de evaluación YOLO',
    'JSON del análisis',
    '/quick-report.pdf',
):
    if required.lower() == 'gravedad no determinada':
        assert 'gravedad no determinada' in html, required
    else:
        assert required in html, required
assert html.count('id="exteriorFile"') == 1
assert html.count('id="interiorFile"') == 1

with tempfile.NamedTemporaryFile(mode="w", suffix=".js", encoding="utf-8") as temporary:
    temporary.write(script)
    temporary.flush()
    subprocess.run(["node","--check",temporary.name],check=True)
print("Dashboard: JavaScript válido y elementos requeridos presentes.")
