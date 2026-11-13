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
with tempfile.NamedTemporaryFile(mode="w", suffix=".js", encoding="utf-8") as temporary:
    temporary.write(script)
    temporary.flush()
    subprocess.run(["node","--check",temporary.name],check=True)
print("Dashboard: JavaScript válido y elementos requeridos presentes.")
