"""Experimental YOLO paint-defect detector (CPU ONNX). Off by default.

Do not enable without an accepted held-out evaluation, a SHA256-pinned release,
and field inspection for false positives on Peruvian housing.
"""
import hashlib
import os
from pathlib import Path
import threading
import urllib.request
import numpy as np
from PIL import Image,ImageDraw
from inspection_taxonomy import CLASSES

MODEL="pintura-deteriorada-candidate.onnx"
URL="https://github.com/Yosep646/tasacion-banco/releases/download/v0.3-paint-experimental/"+MODEL
_lock=threading.Lock()
_session=None


def get_model():
    expected=os.getenv("YOLO_PAINT_SHA256","").strip().lower()
    if len(expected)!=64 or any(c not in "0123456789abcdef" for c in expected):
        raise RuntimeError("Modelo de pintura sin hash auditado")
    modeldir=Path(os.getenv("TASACION_MODEL_DIR","/tmp/tasacion-models"))
    modeldir.mkdir(parents=True,exist_ok=True)
    file=modeldir/MODEL
    if file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest()==expected:
        return file
    part=file.with_suffix(".download")
    try:
        with urllib.request.urlopen(URL,timeout=90) as remote,part.open("wb") as out:
            digest=hashlib.sha256()
            total=0
            while chunk:=remote.read(1024*1024):
                total+=len(chunk)
                if total>75*1024*1024:raise ValueError("Modelo demasiado grande")
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest()!=expected:
            raise ValueError("SHA256 del ONNX no coincide con publicación validada")
        part.replace(file)
    finally:
        part.unlink(missing_ok=True)
    return file


def session():
    global _session
    with _lock:
        if _session is None:
            import onnxruntime as ort
            options=ort.SessionOptions()
            options.intra_op_num_threads=1
            options.inter_op_num_threads=1
            _session=ort.InferenceSession(str(get_model()),sess_options=options,
                                          providers=["CPUExecutionProvider"])
        return _session


def _nms(candidates,threshold=.42):
    keep=[]
    for box,confidence in sorted(candidates,key=lambda x:-x[1]):
        collide=False
        for previous,_ in keep:
            x1,y1,x2,y2=max(box[0],previous[0]),max(box[1],previous[1]),min(box[2],previous[2]),min(box[3],previous[3])
            intersection=max(0,x2-x1)*max(0,y2-y1)
            area1=(box[2]-box[0])*(box[3]-box[1])
            area2=(previous[2]-previous[0])*(previous[3]-previous[1])
            if intersection/max(area1+area2-intersection,1e-6)>threshold:
                collide=True;break
        if not collide:keep.append((box,confidence))
        if len(keep)>=15:break
    return keep


def detect_paint(image_path,threshold=.50):
    model=session()
    with Image.open(image_path) as original:
        rgb=original.convert("RGB")
    w,h=rgb.size
    spec=model.get_inputs()[0]
    shape=spec.shape
    if len(shape)!=4:raise RuntimeError("Entrada ONNX no compatible")
    size=int(shape[2]) if isinstance(shape[2],int) else 416
    if size<320 or size>1280:raise RuntimeError("Modelo no aprobado para esta resolución")
    scale=min(size/w,size/h)
    rw,rh=max(1,round(w*scale)),max(1,round(h*scale))
    px,py=(size-rw)//2,(size-rh)//2
    canvas=Image.new("RGB",(size,size),(114,114,114))
    canvas.paste(rgb.resize((rw,rh),Image.Resampling.BILINEAR),(px,py))
    input_array=np.asarray(canvas,dtype=np.float32).transpose(2,0,1)[None]/255.
    raw=np.asarray(model.run(None,{spec.name:input_array})[0])
    if raw.ndim==3:raw=raw[0]
    if raw.ndim!=2:raise RuntimeError("Salida ONNX no compatible")
    candidates=[]
    if raw.shape[-1]==6:
        iterator=[(row[:4],float(row[4]),True) for row in raw if int(row[5])==0]
    elif raw.shape[0]==5:
        raw=raw.T
        iterator=[(row[:4],float(row[4]),False) for row in raw]
    elif raw.shape[1]==5:
        iterator=[(row[:4],float(row[4]),False) for row in raw]
    else:
        raise RuntimeError("El modelo publicado no es YOLO de una clase")
    for vals,confidence,end_to_end in iterator:
        if not np.isfinite(confidence) or confidence<threshold:continue
        x,y,x2,y2=map(float,vals)
        if not end_to_end:
            x,y,x2,y2=x-x2/2,y-y2/2,x+x2/2,y+y2/2
        points=[max(0,min(w,(x-px)/scale)),max(0,min(h,(y-py)/scale)),
                max(0,min(w,(x2-px)/scale)),max(0,min(h,(y2-py)/scale))]
        if not all(np.isfinite(points)) or points[2]<=points[0] or points[3]<=points[1]:
            continue
        candidates.append((points,confidence))
    results=[]
    overlay=rgb.copy()
    draw=ImageDraw.Draw(overlay)
    for box,confidence in _nms(candidates):
        results.append({
            "category":"pintura_deteriorada",
            "source_label":"paint_defect",
            "label":CLASSES["pintura_deteriorada"]["label"],
            "confidence":round(confidence,4),
            "box":[round(v,2) for v in box],
            "origin":"yolo_public_paint_experimental",
            "severity":"no_determinada",
            "description":"Posible deterioro de pintura, modelo experimental; necesita verificación.",
            "improvement":CLASSES["pintura_deteriorada"]["improvement"],
        })
        draw.rectangle(tuple(box),outline="#e99b22",width=max(2,int(w/350)))
        draw.text((box[0],max(0,box[1]-13)),"Posible daño pintura "+str(round(confidence*100))+"%",fill="#b96707")
    return results,overlay
