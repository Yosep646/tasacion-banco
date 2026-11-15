#!/usr/bin/env python3
"""Exactamente 4 capturas seleccionadas del frontend original de tasacion-banco."""
from pathlib import Path
import io
import re

from PIL import ImageDraw
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import generar_pdf_capturas_codigo as gen

SRC=gen.ROOT/"static"/"index.html"
OUT=gen.ROOT/"capturas-cuatro-hojas-tasacion.pdf"
source=SRC.read_text(encoding="utf-8").splitlines()
# Each selection uses ORIGINAL file line numbers (1-indexed).
PAGES=[
  ("1 / 4   INTERFAZ Y FORMULARIO DE FOTOGRAFIAS",
   [(538,547),(630,645)],
   "Estructura de la pagina, ingreso, dashboard y seleccion de fotos."),
  ("2 / 4   CAMARA Y CAPTURA DE IMAGENES",
   [(580,617)],
   "Acceso a camara, vista previa, captura JPG y cierre del dispositivo."),
  ("3 / 4   ANALISIS IA Y CONSULTA DE PROGRESO",
   [(725,771)],
   "Envio de fotos al backend y seguimiento asincrono del analisis."),
  ("4 / 4   RESULTADOS, TASACION Y DESCARGA PDF",
   [(794,812),(829,844)],
   "Resumen economico, hallazgos y boton para descargar el informe.")
]

def selected_rows(ranges):
    rows=[]
    for i,(first,last) in enumerate(ranges):
        if i:
            rows.append((None,"# ... se omiten aqui lineas intermedias del mismo archivo ...",False))
        for no in range(first,last+1):
            original=source[no-1].expandtabs(4)
            original=re.sub(
                r"sk-(?:proj-)?[A-Za-z0-9_-]{25,}",
                "[CLAVE PRIVADA OCULTA]",original)
            # Long source lines in this HTML are single-line template strings.
            # Abbreviate visibly to retain readable, important lines on 4 pages.
            if len(original)>164:
                original=original[:158]+"  ... [abreviado]"
            for j in range(0,max(1,len(original)),gen.MAX_COLS):
                rows.append((no if j==0 else None,original[j:j+gen.MAX_COLS],j>0))
    return rows

def main():
    assert len(PAGES)==4
    doc=canvas.Canvas(str(OUT),pagesize=(gen.PAPER_W,gen.PAPER_H),pageCompression=1)
    doc.setTitle("4 capturas de codigo web - Tasacion Visual")
    doc.setAuthor("Documentacion de fuente: Yosep646/tasacion-banco")
    for n,(title,ranges,explain) in enumerate(PAGES,1):
        rows=selected_rows(ranges)
        assert len(rows)<=gen.MAX_ROWS,(title,len(rows))
        img=gen.draw_code_screenshot("static/index.html",rows,n,len(PAGES),n)
        draw=ImageDraw.Draw(img)
        draw.rectangle((0,109,gen.WIDTH,144),fill=gen.BG)
        draw.text((28,113),title,font=gen.ui_small,fill=gen.BLUE)
        draw.text((31,gen.HEIGHT-129),explain,font=gen.ui_small,fill=gen.WHITE)
        draw.text((31,gen.HEIGHT-105),
                  "Fragmentos reales del repositorio; lineas largas marcadas como abreviadas.",
                  font=gen.ui_small,fill=gen.GRAY)
        buf=io.BytesIO()
        img.save(buf,"JPEG",quality=88,optimize=True,subsampling=0)
        buf.seek(0)
        doc.drawImage(ImageReader(buf),0,0,width=gen.PAPER_W,height=gen.PAPER_H)
        doc.showPage()
        print(f"HOJA {n}: {title} / {len(rows)} lineas visuales",flush=True)
    doc.save()
    print(f"PDF 4 HOJAS: {OUT} ({OUT.stat().st_size} bytes)",flush=True)

if __name__=="__main__":
    main()
