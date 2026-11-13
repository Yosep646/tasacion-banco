"""ONNX-only detection of 5 visible building defects.

Third-party model IAAC MaAI YOLO26n, AGPL-3.0, 2026.
It has poor reported AP on some surface categories; results require human review.
"""
from pathlib import Path
import hashlib
import os
import threading
import urllib.request
import numpy as np
from PIL import Image,ImageDraw
from inspection_taxonomy import PRETRAINED_5, CLASSES

ASSET_URL="https://github.com/Yosep646/tasacion-banco/releases/download/v0.2-defect5-onnx/"
ASSET="vivienda-defectos-5.onnx"
PINNED_SHA256="849148feb9a15a5b1d02ae5477befe3248f35f9305bd1e188a9035000d69397a"
LABELS=["crack","mold","peeling_paint","stairstep_crack","water_seepage"]
_sessions={}
_lock=threading.Lock()


def model_path():
    return Path(os.getenv("TASACION_MODEL_DIR","/tmp/tasacion-models"))/ASSET


def download_model():
    path=model_path()
    path.parent.mkdir(parents=True,exist_ok=True)
    sha_path=path.with_suffix(".onnx.sha256")
    if not sha_path.is_file():
        with urllib.request.urlopen(ASSET_URL+ASSET+".sha256",timeout=20) as response:
            text=response.read(200).decode("ascii").strip()
        digest=text.split()[0].lower()
        if len(digest)!=64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise RuntimeError("Checksum inválido")
        if digest!=PINNED_SHA256:
            raise RuntimeError("La versión publicada difiere del modelo ONNX auditado")
        sha_path.write_text(digest)
    expected=sha_path.read_text().strip().split()[0].lower()
    if expected!=PINNED_SHA256:
        raise RuntimeError("Modelo local no coincide con el hash aprobado")
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected:
        return path
    part=path.with_suffix(".part")
    try:
        with urllib.request.urlopen(ASSET_URL+ASSET,timeout=85) as response,part.open("wb") as out:
            size=0
            digest=hashlib.sha256()
            while chunk:=response.read(1024*1024):
                size+=len(chunk)
                if size>45_000_000:raise RuntimeError("Modelo demasiado grande")
                out.write(chunk)
                digest.update(chunk)
        if digest.hexdigest()!=expected:raise RuntimeError("SHA256 no coincide")
        part.replace(path)
    finally:part.unlink(missing_ok=True)
    return path


def session():
    with _lock:
        if "model" not in _sessions:
            import onnxruntime as ort
            opts=ort.SessionOptions()
            opts.intra_op_num_threads=1
            opts.inter_op_num_threads=1
            _sessions["model"]=ort.InferenceSession(str(download_model()),sess_options=opts,providers=["CPUExecutionProvider"])
    return _sessions["model"]


def unpack_predictions(raw):
    """Handle YOLO detect export (5+4,N) or end-to-end (N,6)."""
    out=np.asarray(raw)
    if out.ndim==3:out=out[0]
    if out.ndim!=2:raise RuntimeError("Salida del modelo desconocida")
    # End-to-end: x1,y1,x2,y2,confidence,class
    if out.shape[1]==6 and out.shape[0]!=6:
        return [(list(map(float,row[:4])),float(row[4]),int(row[5]),"xyxy") for row in out]
    if out.shape[0]==6 and out.shape[1]!=6:
        out=out.T
        return [(list(map(float,row[:4])),float(row[4]),int(row[5]),"xyxy") for row in out]
    if out.shape[0]==9:out=out.T
    if out.shape[1]==9:
        rows=[]
        for row in out:
            cls=int(np.argmax(row[4:9]))
            rows.append((list(map(float,row[:4])),float(row[4+cls]),cls,"xywh"))
        return rows
    raise RuntimeError("Salida ONNX incompatible con 5 clases")


def detect_defects(image_path):
    model=session()
    with Image.open(image_path) as image:
        im=image.convert("RGB")
    w,h=im.size
    ratio=min(640/w,640/h)
    rw,rh=max(1,round(w*ratio)),max(1,round(h*ratio))
    px,py=(640-rw)//2,(640-rh)//2
    inp=Image.new("RGB",(640,640),(114,114,114))
    inp.paste(im.resize((rw,rh),Image.Resampling.BILINEAR),(px,py))
    tensor=np.asarray(inp,dtype=np.float32).transpose(2,0,1)[None]/255.
    raw=model.run(None,{model.get_inputs()[0].name:tensor})[0]
    rows=unpack_predictions(raw)
    thresholds={"crack":0.35,"stairstep_crack":0.35,"mold":0.5,"peeling_paint":0.55,"water_seepage":0.5}
    candidates=[]
    for box,conf,cls,mode in rows:
        if cls<0 or cls>=len(LABELS):continue
        label=LABELS[cls]
        if not np.isfinite(conf) or conf<thresholds[label]:continue
        if mode=="xywh":
            cx,cy,bw,bh=box
            box=[cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2]
        xmin,ymin,xmax,ymax=box
        if not all(np.isfinite(v) for v in box):continue
        scaled=[max(0,min(w,(xmin-px)/ratio)),max(0,min(h,(ymin-py)/ratio)),
                max(0,min(w,(xmax-px)/ratio)),max(0,min(h,(ymax-py)/ratio))]
        if scaled[2]<=scaled[0] or scaled[3]<=scaled[1]:continue
        candidates.append((scaled,conf,label))
    candidates.sort(key=lambda x:x[1],reverse=True)
    results=[]
    for box,conf,label in candidates[:150]:
        overlap=False
        for prev in results:
            if prev["source_label"]!=label:continue
            old=prev["box"]
            xx1,yy1,xx2,yy2=max(box[0],old[0]),max(box[1],old[1]),min(box[2],old[2]),min(box[3],old[3])
            intersect=max(0,xx2-xx1)*max(0,yy2-yy1)
            a=(box[2]-box[0])*(box[3]-box[1])
            b=(old[2]-old[0])*(old[3]-old[1])
            if intersect/max(a+b-intersect,1e-6)>.45:overlap=True;break
        if overlap:continue
        category=PRETRAINED_5[label]
        results.append({
            "category":category,"label":CLASSES[category]["label"],
            "source_label":label,"confidence":round(float(conf),4),
            "box":[round(float(v),2) for v in box],
            "improvement":CLASSES[category]["improvement"],
            "severity":"no_determinada",
            "description":"Posible "+CLASSES[category]["label"].lower()+" detectada; verificar presencialmente.",
            "origin":"yolo26_defect5",
            "model":"IAAC-MaAI YOLO26n, AGPL-3.0",
        })
        if len(results)>=25:break
    overlay=im.copy()
    draw=ImageDraw.Draw(overlay)
    colors={"crack":"#dc3545","stairstep_crack":"#d13026","mold":"#7d51cb","peeling_paint":"#dd9234","water_seepage":"#157ec5"}
    for d in results:
        coords=tuple(d["box"])
        draw.rectangle(coords,outline=colors[d["source_label"]],width=max(2,int(w/350)))
        draw.text((coords[0],max(0,coords[1]-14)),d["source_label"]+" "+str(round(d["confidence"]*100))+"%",
                  fill=colors[d["source_label"]])
    return results,overlay
