"""Export the pinned public defect YOLO26n checkpoint to an ONNX CPU artifact.

Third-party .pt files can execute code on import: use only after independently
reviewing their provenance. The published weights are AGPL-3.0.
"""
from pathlib import Path
import hashlib
import urllib.request

URL="https://huggingface.co/iaac-maai/day4-team1-building-defect-detector/resolve/main/model.pt"
DEST=Path("weights/defect-five.pt")
DEST.parent.mkdir(exist_ok=True,parents=True)
def download():
    if DEST.exists() and DEST.stat().st_size > 1_000_000:
        return DEST
    temp=DEST.with_suffix(".partial")
    try:
        with urllib.request.urlopen(URL,timeout=90) as response,temp.open("wb") as dest:
            total=0
            while block:=response.read(1024*1024):
                total+=len(block)
                if total>30_000_000:raise RuntimeError("Unexpectedly large checkpoint")
                dest.write(block)
        temp.replace(DEST)
    finally:
        temp.unlink(missing_ok=True)
    return DEST

if __name__=="__main__":
    from ultralytics import YOLO
    file=download()
    print("Original SHA256:",hashlib.sha256(file.read_bytes()).hexdigest())
    model=YOLO(str(file))
    print("Task:",model.task,"Labels:",model.names)
    assert model.task=="detect"
    assert len(model.names)==5
    exported=Path(model.export(format="onnx",imgsz=640,batch=1,simplify=False,opset=17,dynamic=False))
    destination=Path("weights/vivienda-defectos-5.onnx")
    destination.write_bytes(exported.read_bytes())
    sha=hashlib.sha256(destination.read_bytes()).hexdigest()
    (destination.parent/(destination.name+".sha256")).write_text(sha+"  "+destination.name+"\n")
    print("Exported:",destination,"bytes:",destination.stat().st_size,"SHA256:",sha)
