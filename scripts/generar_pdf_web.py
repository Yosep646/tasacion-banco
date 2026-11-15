#!/usr/bin/env python3
"""PDF de capturas del archivo HTML/CSS/JavaScript de la web Tasacion Visual."""
from pathlib import Path
import generar_pdf_capturas_codigo as gen

gen.OUT = gen.ROOT / "capturas-codigo-web-tasacion.pdf"
gen.source_files = lambda: [gen.ROOT / "static/index.html"]

def draw_cover(draw, files, total_lines, commit):
    draw.rectangle((0, 0, gen.WIDTH, gen.HEIGHT), fill=gen.BG)
    draw.rounded_rectangle((90, 170, gen.WIDTH-90, gen.HEIGHT-180), radius=38, fill=gen.PANEL)
    draw.rectangle((90, 170, 104, gen.HEIGHT-180), fill=gen.BLUE)
    draw.text((150, 255), "WEB DE TASACION", font=gen.ui_large, fill=gen.WHITE)
    draw.text((150, 340), "CAPTURAS DEL CODIGO DE LA WEB", font=gen.ui, fill=gen.BLUE)
    draw.text((150, 490), "Archivo: static/index.html", font=gen.ui, fill=gen.WHITE)
    draw.text((150, 560), "Tecnologias: HTML, CSS y JavaScript", font=gen.ui, fill=gen.WHITE)
    draw.text((150, 630), "Repositorio: Yosep646/tasacion-banco", font=gen.ui, fill=gen.WHITE)
    draw.text((150, 700), "Commit: "+commit[:12], font=gen.ui, fill=gen.GRAY)
    draw.text((150, 780), f"Lineas de codigo: {total_lines:,}", font=gen.ui, fill=gen.WHITE)
    yy = 920
    for line in [
        "Incluye TODAS las lineas del codigo de la interfaz:",
        "diseno, estilos CSS, formularios, camara,",
        "JavaScript, analisis y resultados de tasacion.",
        "Capturas visuales tipo editor, con numero de linea.",
        "El backend Python se encuentra en el PDF general.",
    ]:
        draw.text((150, yy), line, font=gen.ui_small, fill=gen.WHITE)
        yy += 46
    draw.text((150, gen.HEIGHT-305), "Version original de GitHub - sin claves privadas", font=gen.ui_small, fill=gen.GRAY)

gen.draw_cover = draw_cover
gen.main()
