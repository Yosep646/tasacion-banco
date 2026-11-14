"""Semantic walls/floor/ceiling segmentation: no cloud call or pretrained downloads."""
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

import app
import surface_segmentation as surfaces


class FakeSession:
    def get_inputs(self):
        return [SimpleNamespace(name="pixel_values")]

    def run(self, unused, feeds):
        pixels=feeds["pixel_values"]
        assert pixels.shape == (1, 3, 512, 512)
        logits=np.full((1,150,8,8), -3, dtype=np.float32)
        logits[:,0,:,:]=3
        logits[:,3,4:,:]=5
        logits[:,5,:2,:]=7
        return [logits]


def test_surface_semantic_masks_and_render(monkeypatch,tmp_path):
    monkeypatch.setattr(surfaces,"get_session",lambda: FakeSession())
    path=tmp_path/"room.jpg"
    Image.new("RGB",(120,80),"white").save(path)
    meta,overlay,mask=surfaces.segment_surfaces(path)
    assert overlay.size==mask.size==(120,80)
    assert mask.mode=="RGBA"
    assert meta["status"]=="preliminar_por_verificar"
    assert [c["key"] for c in meta["classes"]]==["pared","piso","techo"]
    parts={c["key"]:c for c in meta["classes"]}
    assert parts["techo"]["pixels"]>0
    assert parts["piso"]["pixels"]>0
    assert parts["pared"]["pixels"]>0
    assert 99.9 <= sum(c["percentage_image"] for c in meta["classes"]) <= 100.1
    assert meta["coverage_unit"]=="porcentaje_de_pixeles_de_la_fotografia"
    assert overlay.getpixel((60,5))!=(255,255,255)


def test_unsupported_logits_are_rejected():
    import pytest
    with pytest.raises(RuntimeError):
        surfaces.render_surfaces(Image.new("RGB",(16,16)),np.zeros((1,3,4,4),dtype=np.float32))


def test_segmentation_api_requires_flag_and_persists(monkeypatch):
    client=TestClient(app.app)
    r=client.post("/api/login",json={
        "username":"administrador","password":"strong-test-password"})
    assert r.status_code==200,r.text
    cid=client.post("/api/cases",json={
        "asset_type":"vivienda","title":"Prueba segmentación"}).json()["id"]
    picture=io.BytesIO()
    Image.new("RGB",(80,80),"white").save(picture,format="JPEG")
    response=client.post("/api/cases/"+cid+"/evidence",data={"shot":"Interior"},
                         files={"file":("interior.jpg",picture.getvalue(),"image/jpeg")})
    assert response.status_code==201,response.text
    eid=response.json()["id"]
    monkeypatch.setenv("SURFACE_SEGMENTATION_ENABLED","0")
    assert client.post(f"/api/cases/{cid}/segment-surfaces/{eid}").status_code==503
    monkeypatch.setenv("SURFACE_SEGMENTATION_ENABLED","1")
    monkeypatch.setattr(surfaces,"get_session",lambda: FakeSession())
    r=client.post(f"/api/cases/{cid}/segment-surfaces/{eid}")
    assert r.status_code==200,r.text
    assert r.json()["result"]["surfaces"]["classes"][0]["key"]=="pared"
    assert client.get(f"/api/evidence/{eid}/surface-overlay").status_code==200
    mask=client.get(f"/api/evidence/{eid}/surface-mask")
    assert mask.status_code==200 and mask.headers["content-type"]=="image/png"
    stored=client.get(f"/api/cases/{cid}").json()["visual_assessments"]
    assert any("segformer_b0_ade20k_onnx" in row["result_json"] for row in stored)
    assert all(row["state"]=="pendiente" for row in stored)
