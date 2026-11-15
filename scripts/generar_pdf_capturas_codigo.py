#!/usr/bin/env python3
"""Generate a complete PDF of screenshot-style source-code pages.

Runs inside GitHub Actions against the exact checked-out repository. A screenshot
is rendered for every code-page chunk using Pillow and then embedded in a PDF.
No production credentials are needed or downloaded.
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import get_lexer_for_filename, TextLexer
from pygments.token import Comment, Error, Keyword, Literal, Name, Operator, Punctuation, String
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "capturas-codigo-fuente-tasacion.pdf"
EXCLUDE_DIR = {".git", "__pycache__", ".venv", ".pytest_cache", "node_modules", "data", "runs"}
EXCLUDE_EXT = {".webp", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".pt", ".onnx", ".pyc", ".db", ".sqlite3"}
CODE_EXT = {".py", ".html", ".css", ".js", ".json", ".yaml", ".yml", ".md", ".txt", ".toml", ".ini", ".cfg", ".sh"}
INCLUDE_NAMES = {".gitignore", ".env.example", "Dockerfile", "Makefile"}
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_UI = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
WIDTH, HEIGHT = 1240, 1680
PAPER_W, PAPER_H = 595.28, 841.89
LEFT = 75
TOP = 149
LINE_H = 25
MAX_COLS = 93
MAX_ROWS = 56
RED = (236, 101, 113)
GREEN = (155, 206, 154)
BLUE = (117, 177, 246)
YELLOW = (238, 199, 116)
PURPLE = (195, 142, 238)
WHITE = (219, 226, 235)
GRAY = (123, 138, 158)
BG = (20, 25, 37)
BAR = (31, 39, 54)
PANEL = (25, 32, 46)

font = ImageFont.truetype(FONT_PATH, 18)
bold = ImageFont.truetype(FONT_BOLD, 20)
ui = ImageFont.truetype(FONT_UI, 20)
ui_large = ImageFont.truetype(FONT_UI, 34)
ui_small = ImageFont.truetype(FONT_UI, 17)
char_width = int(ImageDraw.Draw(Image.new("RGB", (40, 40))).textlength("M", font=font))


def source_files():
    paths = []
    try:
        tracked = subprocess.check_output(
            ["git", "ls-files"], cwd=ROOT, text=True
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        tracked = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()]
    for rel in tracked:
        rel = rel.replace("\\", "/")
        parts = Path(rel).parts
        if any(p in EXCLUDE_DIR for p in parts) or rel.endswith(".pdf"):
            continue
        path = ROOT / rel
        if not path.is_file() or path.suffix.lower() in EXCLUDE_EXT:
            continue
        if path.suffix.lower() not in CODE_EXT and path.name not in INCLUDE_NAMES:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        paths.append(path)
    return sorted(set(paths), key=lambda x: str(x.relative_to(ROOT)).casefold())


def token_color(token):
    if token in Comment:
        return GRAY
    if token in Keyword:
        return PURPLE
    if token in String or token in Literal.String:
        return GREEN
    if token in Literal.Number:
        return YELLOW
    if token in Name.Function or token in Name.Class:
        return BLUE
    if token in Name.Builtin:
        return YELLOW
    if token in Operator or token in Punctuation:
        return WHITE
    if token in Error:
        return RED
    return WHITE


def syntax_line(line, filename):
    try:
        lexer = get_lexer_for_filename(filename, stripnl=False)
    except Exception:
        lexer = TextLexer(stripnl=False)
    result = []
    for token, value in lex(line, lexer):
        if value:
            result.extend((ch, token_color(token)) for ch in value if ch not in "\r\n")
    return result


SECRET = re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{25,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")


def chunks(path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    if not lines:
        lines = ["(archivo vacío)"]
    pages = []
    rows = []
    start = 1
    for line_no, original in enumerate(lines, start=1):
        expanded = original.expandtabs(4)
        expanded = SECRET.sub("[CLAVE PRIVADA OCULTA]", expanded)
        for offset in range(0, max(1, len(expanded)), MAX_COLS):
            segment = expanded[offset:offset+MAX_COLS]
            if len(rows) >= MAX_ROWS:
                pages.append((start, rows))
                rows = []
                start = line_no
            rows.append((line_no if offset == 0 else None, segment, offset > 0))
    if rows:
        pages.append((start, rows))
    return pages, len(lines)


def draw_cover(draw, files, total_lines, commit):
    draw.rectangle((0, 0, WIDTH, HEIGHT), fill=BG)
    draw.rounded_rectangle((90, 170, WIDTH-90, HEIGHT-180), radius=38, fill=PANEL)
    draw.rectangle((90, 170, 104, HEIGHT-180), fill=BLUE)
    draw.text((150, 250), "WEB DE TASACIÓN", font=ui_large, fill=WHITE)
    draw.text((150, 330), "CAPTURAS DEL CÓDIGO FUENTE", font=ui, fill=BLUE)
    draw.text((150, 475), "Repositorio: Yosep646/tasacion-banco", font=ui, fill=WHITE)
    draw.text((150, 525), "Rama: main (versión verificada)", font=ui, fill=WHITE)
    draw.text((150, 575), "Commit: "+commit[:12], font=ui, fill=GRAY)
    draw.text((150, 635), f"Archivos de texto: {files}", font=ui, fill=WHITE)
    draw.text((150, 685), f"Líneas de código/documentación: {total_lines:,}", font=ui, fill=WHITE)
    draw.text((150, 790), "Documento visual de referencia", font=ui, fill=YELLOW)
    y=846
    for line in [
        "Incluye todos los archivos de código y configuración",
        "versionados en el repositorio, con números de línea.",
        "Las imágenes binarias y los pesos de IA no son código.",
        "Cualquier clave privada detectada se oculta.",
    ]:
        draw.text((150, y), line, font=ui_small, fill=WHITE)
        y+=42
    draw.text((150, HEIGHT-305), "Fuente: github.com/Yosep646/tasacion-banco", font=ui_small, fill=GRAY)


def draw_code_screenshot(rel, chunk, part, total, global_page):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, WIDTH, 68), fill=BAR)
    draw.ellipse((25, 25, 43, 43), fill=RED)
    draw.ellipse((55, 25, 73, 43), fill=YELLOW)
    draw.ellipse((85, 25, 103, 43), fill=GREEN)
    short_rel = rel if len(rel) <= 85 else "..." + rel[-82:]
    draw.text((130, 24), short_rel, font=bold, fill=WHITE)
    draw.rectangle((0, 68, WIDTH, 105), fill=PANEL)
    subtitle=f"WEB DE TASACION  /  {rel}  /  fragmento {part}/{total}"
    if len(subtitle)>103:subtitle=subtitle[:100]+"..."
    draw.text((33, 78), subtitle, font=ui_small, fill=GRAY)
    draw.rectangle((0, 105, 66, HEIGHT-87), fill=(22, 29, 43))
    for rowidx, (line_no, text, wrapped) in enumerate(chunk):
        y=TOP+rowidx*LINE_H
        if line_no is not None:
            draw.text((12,y),f"{line_no:4d}",font=font,fill=GRAY)
        if wrapped:
            draw.text((65,y),"↳",font=font,fill=GRAY)
        x=LEFT
        for ch, color in syntax_line(text, rel):
            if x+char_width>WIDTH-18:
                break
            if ch!=" ":
                draw.text((x,y),ch,font=font,fill=color)
            x+=char_width
    draw.rectangle((0, HEIGHT-88, WIDTH, HEIGHT), fill=BAR)
    draw.text((30,HEIGHT-64), "CÓDIGO FUENTE - CAPTURA VISUAL", font=ui_small, fill=GRAY)
    draw.text((WIDTH-235,HEIGHT-64),f"Página {global_page}",font=ui_small,fill=BLUE)
    return img


def main():
    files=source_files()
    contents=[(p, *chunks(p)) for p in files]
    total_lines=sum(line_count for _,_,line_count in contents)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    pdf=canvas.Canvas(str(OUT),pagesize=(PAPER_W,PAPER_H),pageCompression=1)
    pdf.setTitle("Capturas de todo el código fuente - Tasación Visual")
    pdf.setAuthor("Documentación del repositorio Yosep646/tasacion-banco")
    cover=Image.new("RGB",(WIDTH,HEIGHT))
    draw_cover(ImageDraw.Draw(cover),len(files),total_lines,commit)
    pdf.drawImage(ImageReader(cover),0,0,width=PAPER_W,height=PAPER_H)
    pdf.showPage()
    count=1
    for path, parts, nlines in contents:
        rel=path.relative_to(ROOT).as_posix()
        for idx, (_,chunk) in enumerate(parts,start=1):
            count+=1
            img=draw_code_screenshot(rel,chunk,idx,len(parts),count)
            stream=io.BytesIO()
            img.save(stream,"JPEG",quality=84,optimize=True,subsampling=0)
            stream.seek(0)
            pdf.drawImage(ImageReader(stream),0,0,width=PAPER_W,height=PAPER_H)
            pdf.bookmarkPage(f"source_{count}")
            if idx==1:
                pdf.addOutlineEntry(rel,f"source_{count}",level=0,closed=False)
            pdf.showPage()
        print(f"ARCHIVO {rel}: {nlines} líneas, {len(parts)} capturas",flush=True)
    pdf.save()
    print(f"PDF FINAL: {OUT}  ({count} páginas, {len(files)} archivos, {total_lines} líneas, {OUT.stat().st_size} bytes)",flush=True)


if __name__=="__main__":
    main()
