"""Gate experimental ONNX release on independently tested public paint dataset."""
import hashlib,json,sys
from pathlib import Path

path=Path(sys.argv[1]) if len(sys.argv)>1 else Path("paint-artifact")
metrics=json.loads((path/"metrics.json").read_text(encoding="utf-8"))
provenance=json.loads((path/"data"/"provenance.json").read_text(encoding="utf-8"))
if not metrics.get("technical_gate_passed"):
    raise SystemExit("BLOCKED: Model did not pass technical gate")
if metrics.get("ap50",0)<.40 or metrics.get("recall",0)<.50:
    raise SystemExit("BLOCKED: insufficient AP50 / recall")
if provenance.get("license")!="CC BY 4.0" or provenance.get("source")!="https://huggingface.co/datasets/imaadd05/paint-defects-roboflow-main":
    raise SystemExit("BLOCKED: invalid provenance")
if len(provenance.get("image_hashes_by_split",{}))!=3:
    raise SystemExit("BLOCKED: no independent test split")
model=path/"pintura-deteriorada-candidate.onnx"
if not model.is_file() or not 100_000<=model.stat().st_size<=100_000_000:
    raise SystemExit("BLOCKED: missing/invalid ONNX model")
import onnx
import onnxruntime as ort
import numpy as np
onnx.checker.check_model(str(model))
options=ort.SessionOptions()
options.intra_op_num_threads=1
session=ort.InferenceSession(str(model),sess_options=options,providers=["CPUExecutionProvider"])
spec=session.get_inputs()[0]
shape=[int(v) if isinstance(v,int) and v>0 else (1 if i==0 else 416) for i,v in enumerate(spec.shape)]
if len(shape)!=4 or shape[0]!=1 or shape[1]!=3:
    raise SystemExit("BLOCKED: unsupported input shape")
outputs=session.run(None,{spec.name:np.zeros(shape,dtype=np.float32)})
if not outputs or not np.isfinite(outputs[0]).all():
    raise SystemExit("BLOCKED: invalid CPU inference")
sha=hashlib.sha256(model.read_bytes()).hexdigest()
(path/"model.sha256").write_text(sha+"  pintura-deteriorada-candidate.onnx\n")
(path/"publication_report.json").write_text(json.dumps({
    "name":"YOLO experimental, paint defects only",
    "sha256":sha,"metrics":metrics,"dataset":provenance["source"],
    "training_license":"CC BY 4.0","model_license":"AGPL-3.0",
    "status":"experimental_not_authorized_for_automatic_valuation",
    "field_validation":"pending_peruvian_housing",
    "no_credit_decisions":True
},indent=2))
print("PASS: ONNX CPU, metrics and provenance. SHA256:",sha)
