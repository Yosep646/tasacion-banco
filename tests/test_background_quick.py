"""Queued inspections: quick HTTP acceptance, persistent result and safe retries.

No network, real weight downloads or OpenAI credentials needed for unit tests.
"""
import io
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from PIL import Image

import app


def photo():
    data=io.BytesIO()
    Image.new("RGB",(96,72),"white").save(data,"JPEG")
    return data.getvalue()


def login():
    client=TestClient(app.app)
    response=client.post("/api/login",json={
        "username":"administrador","password":"strong-test-password"})
    assert response.status_code==200,response.text
    return client


def fake_surfaces(path):
    with Image.open(path) as im:
        size=im.size
    metadata={
        "status":"preliminar_por_verificar",
        "method":"segformer_b0_ade20k_onnx",
        "classes":[
            {"key":key,"label":name,"percentage_image":value,"color":color}
            for key,name,value,color in (
                ("pared","Pared",50,"#2a7ee1"),
                ("piso","Piso",25,"#199f75"),
                ("techo","Techo",10,"#f29a2e"),
            )
        ],
        "limitations":["Porcentajes de píxeles, no metros cuadrados"],
    }
    return metadata,Image.new("RGB",size,"white"),Image.new("RGBA",size,(0,0,0,0))


def test_background_quick_two_images_complete_with_persisted_report(monkeypatch):
    monkeypatch.setenv("SURFACE_SEGMENTATION_ENABLED","1")
    monkeypatch.setenv("TASACION_LOW_MEMORY_MODE","1")
    monkeypatch.delenv("YOLO_CPU_ENABLED",raising=False)
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    monkeypatch.setattr(app,"segment_surfaces",fake_surfaces)
    client=login()
    response=client.post("/api/quick-inspection/start",files=[
        ("exterior",("front.jpg",photo(),"image/jpeg")),
        ("interior",("room.jpg",photo(),"image/jpeg")),
    ])
    assert response.status_code==202,response.text
    cid=response.json()["case_id"]
    job=client.get(f"/api/quick-inspection/jobs/{cid}")
    assert job.status_code==200,job.text
    result=job.json()
    assert result["status"]=="completed"
    report=result["report"]
    assert report["case_id"]==cid
    assert report["images_analyzed"]==2
    assert len(report["image_summaries"])==2
    assert report["human_review_required"]
    assert report["property_market_value"] is None
    saved=client.get(f"/api/cases/{cid}/quick-report")
    assert saved.status_code==200
    assert saved.json()["images_analyzed"]==2
    for image in report["image_summaries"]:
        eid=image["evidence_id"]
        assert client.get(f"/api/evidence/{eid}/surface-overlay").status_code==200
        assert client.get(f"/api/evidence/{eid}/surface-mask").status_code==200
    pdf=client.get(f"/api/cases/{cid}/quick-report.pdf")
    assert pdf.status_code==200 and pdf.content.startswith(b"%PDF")
    assert client.post(f"/api/quick-inspection/jobs/{cid}/retry").json()["status"]=="completed"


def test_background_quick_retry_after_pipeline_failure(monkeypatch):
    monkeypatch.setenv("SURFACE_SEGMENTATION_ENABLED","1")
    monkeypatch.delenv("YOLO_CPU_ENABLED",raising=False)
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    monkeypatch.setattr(app,"segment_surfaces",fake_surfaces)
    original=app.summarize
    failures=[True]
    def intermittent(*args,**kwargs):
        if failures[0]:
            failures[0]=False
            raise RuntimeError("Simulated interrupted inference")
        return original(*args,**kwargs)
    monkeypatch.setattr(app,"summarize",intermittent)
    client=login()
    response=client.post("/api/quick-inspection/start",files={
        "exterior":("front.jpg",photo(),"image/jpeg")})
    assert response.status_code==202,response.text
    cid=response.json()["case_id"]
    previous=client.get(f"/api/quick-inspection/jobs/{cid}").json()
    assert previous["status"]=="failed"
    assert "Simulated" not in previous["message"]
    retried=client.post(f"/api/quick-inspection/jobs/{cid}/retry")
    assert retried.status_code==202,retried.text
    result=client.get(f"/api/quick-inspection/jobs/{cid}").json()
    assert result["status"]=="completed"
    assert result["report"]["images_analyzed"]==1
    assert len(client.get(f"/api/cases/{cid}").json()["evidence"])==1


def test_queue_status_without_login_or_missing_case():
    unauthorized=TestClient(app.app).get(
        "/api/quick-inspection/jobs/00000000-0000-0000-0000-000000000000")
    assert unauthorized.status_code==401
    client=login()
    response=client.get("/api/quick-inspection/jobs/00000000-0000-0000-0000-000000000000")
    assert response.status_code==404


def test_multimodal_quota_error_does_not_stop_local_two_photo_analysis(monkeypatch):
    """An API 429 must not block two locally analysed photos or create fake GPT findings."""
    from vision_service import VisionProviderError
    monkeypatch.setenv("SURFACE_SEGMENTATION_ENABLED","1")
    monkeypatch.setenv("OPENAI_API_KEY","dummy-secret")
    monkeypatch.setenv("VISION_MULTIMODAL_ENABLED","1")
    monkeypatch.setenv("TASACION_LOW_MEMORY_MODE","1")
    monkeypatch.delenv("YOLO_CPU_ENABLED",raising=False)
    monkeypatch.setattr(app,"segment_surfaces",fake_surfaces)
    attempted=[]
    def unavailable(*args):
        attempted.append(1)
        raise VisionProviderError(
            "rate_limited", "Límite de solicitudes alcanzado",429)
    monkeypatch.setattr(app,"analyze_photo",unavailable)
    client=login()
    response=client.post("/api/quick-inspection/start",files=[
        ("exterior",("front.jpg",photo(),"image/jpeg")),
        ("interior",("room.jpg",photo(),"image/jpeg")),
    ])
    assert response.status_code==202,response.text
    case_id=response.json()["case_id"]
    state=client.get(f"/api/quick-inspection/jobs/{case_id}").json()
    assert state["status"]=="completed"
    report=state["report"]
    assert report["images_analyzed"]==2
    assert len(attempted)==1
    assert report["multimodal_vision"]["status"]=="provider_unavailable"
    assert report["multimodal_vision"]["reason"]=="rate_limited"
    assert report["multimodal_vision"]["images_analyzed_by_multimodal"]==0
    assert all("multimodal_visual" not in name for name in report["models"])
    assert all(item["surfaces"] for item in report["image_summaries"])
    assert report["property_market_value"] is None
    assert report["human_review_required"]


def test_pause_external_vision_but_keep_local_segmentation(monkeypatch):
    monkeypatch.setenv("SURFACE_SEGMENTATION_ENABLED","1")
    monkeypatch.setenv("OPENAI_API_KEY","dummy-secret")
    monkeypatch.setenv("VISION_MULTIMODAL_ENABLED","0")
    monkeypatch.delenv("YOLO_CPU_ENABLED",raising=False)
    monkeypatch.setattr(app,"segment_surfaces",fake_surfaces)
    def should_not_call(*args):
        raise AssertionError("GPT must stay paused for automatic inspection")
    monkeypatch.setattr(app,"analyze_photo",should_not_call)
    client=login()
    readiness=client.get("/api/model-readiness").json()
    assert readiness["multimodal_vision"]["state"]=="paused_quota_limited"
    assert readiness["multimodal_vision"]["automatic_enabled"] is False
    response=client.post("/api/quick-inspection/start",files={
        "exterior":("front.jpg",photo(),"image/jpeg")})
    assert response.status_code==202,response.text
    cid=response.json()["case_id"]
    job=client.get(f"/api/quick-inspection/jobs/{cid}").json()
    assert job["status"]=="completed"
    report=job["report"]
    assert report["multimodal_vision"]["status"]=="paused"
    assert report["multimodal_vision"]["images_analyzed_by_multimodal"]==0
    assert report["image_summaries"][0]["surfaces"]
    assert report["status"]=="preliminar_automatico"
