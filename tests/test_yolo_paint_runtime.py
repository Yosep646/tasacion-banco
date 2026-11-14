"""Mocked ONNX tensor tests; these do not claim real model accuracy."""
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

import yolo_paint

class Session:
    def get_inputs(self):
        return [type("Input",(),{"name":"images","shape":[1,3,416,416]})()]
    def run(self,names,feed):
        assert feed["images"].shape==(1,3,416,416)
        # End-to-end model: x1,y1,x2,y2,confidence,class
        return [np.array([[[30.,50.,200.,260.,.85,0.],
                           [31.,51.,199.,259.,.78,0.],
                           [50.,80.,90.,100.,.95,1.]]],dtype=np.float32)]

def test_paint_cpu_postprocessor(monkeypatch):
    monkeypatch.setattr(yolo_paint,"session",lambda:Session())
    with tempfile.TemporaryDirectory() as folder:
        image=Path(folder)/"paint.jpg"
        Image.new("RGB",(416,416),"#e6e3da").save(image)
        result,annotated=yolo_paint.detect_paint(str(image))
        assert len(result)==1
        assert result[0]["category"]=="pintura_deteriorada"
        assert result[0]["confidence"]>0.8
        assert annotated.size==(416,416)

def test_no_paint_model_without_pinned_hash(monkeypatch):
    monkeypatch.delenv("YOLO_PAINT_SHA256",raising=False)
    with pytest.raises(RuntimeError,match="hash auditado"):
        yolo_paint.get_model()
