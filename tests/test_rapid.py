"""One- and two-photo quick inspection: automatic report and safe pricing."""
import io
from PIL import Image
from fastapi.testclient import TestClient

import app
from rapid_service import build_quote, summarize


def photo():
    out=io.BytesIO()
    Image.new("RGB",(80,80),"white").save(out,format="JPEG")
    return out.getvalue()


def client_login():
    client=TestClient(app.app)
    response=client.post("/api/login",json={"username":"administrador",
                                            "password":"strong-test-password"})
    assert response.status_code==200,response.text
    return client


def stub_yolo(path):
    return ([{
        "category":"fisura_visible", "confidence":0.86,
        "description":"Posible fisura detectada automáticamente",
        "improvement":"Revisión presencial",
        "box":[5,6,42,48]
    }],Image.new("RGB",(80,80),"white"))


def test_rapid_one_photo_with_immediate_pdf(monkeypatch):
    monkeypatch.setenv("YOLO_CPU_ENABLED","1")
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    monkeypatch.setattr(app,"predict_cracks",stub_yolo)
    client=client_login()
    response=client.post("/api/quick-inspection",
        files={"exterior":("fachada.jpg",photo(),"image/jpeg")})
    assert response.status_code==201,response.text
    report=response.json()
    assert report["images_analyzed"]==1
    assert len(report["issues"])==1
    assert report["quote"]["total_pen"] is None
    assert report["property_market_value"] is None
    assert report["human_review_required"]
    assert report["status"]=="preliminar_automatico"
    case_id=report["case_id"]
    retrieved=client.get(f"/api/cases/{case_id}/quick-report")
    assert retrieved.status_code==200
    assert retrieved.json()["case_id"]==case_id
    pdf=client.get(f"/api/cases/{case_id}/quick-report.pdf")
    assert pdf.status_code==200
    assert pdf.content.startswith(b"%PDF")
    evidence=report["image_summaries"][0]["evidence_id"]
    assert client.get(f"/api/evidence/{evidence}/yolo-overlay").status_code==200
    case=client.get(f"/api/cases/{case_id}").json()
    assert case["visual_condition"]["condition_score"] is None


def test_rapid_two_photos_and_no_fabricated_pricing(monkeypatch):
    monkeypatch.setenv("YOLO_CPU_ENABLED","1")
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    monkeypatch.setattr(app,"predict_cracks",lambda path:([],Image.new("RGB",(80,80))))
    client=client_login()
    resp=client.post("/api/quick-inspection",files=[
        ("exterior",("outside.jpg",photo(),"image/jpeg")),
        ("interior",("inside.jpg",photo(),"image/jpeg")),
    ])
    assert resp.status_code==201,resp.text
    result=resp.json()
    assert result["images_analyzed"]==2
    assert result["condition"]=="No determinable con las imágenes disponibles"
    assert result["quote"]["total_pen"] is None
    assert result["property_market_value"] is None


def test_invalid_photo_and_model_unavailable(monkeypatch):
    monkeypatch.setenv("YOLO_CPU_ENABLED","1")
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    client=client_login()
    r=client.post("/api/quick-inspection",files={"exterior":("broken.jpg",b"not-an-image","image/jpeg")})
    assert r.status_code==422
    monkeypatch.delenv("YOLO_CPU_ENABLED",raising=False)
    r=client.post("/api/quick-inspection",files={"exterior":("valid.jpg",photo(),"image/jpeg")})
    assert r.status_code==503


def test_catalog_requires_documented_unit_price(monkeypatch):
    issues=[{"category":"fisura_visible"}]
    monkeypatch.setenv("TASACION_PRICE_CATALOG",
        '{"fisura_visible":{"unit_price_pen":120,"unit":"metro lineal","source":"Cotización local fechada"}}')
    quotation=build_quote(issues)
    assert quotation["items"][0]["unit_price_pen"]==120
    assert quotation["items"][0]["quantity"] is None
    assert quotation["total_pen"] is None
    monkeypatch.setenv("TASACION_PRICE_CATALOG",
        '{"fisura_visible":{"unit_price_pen":120,"unit":"metro lineal"}}')
    assert build_quote(issues)["items"]==[]


def test_economic_scenario_is_documented_and_not_inferred(monkeypatch):
    from valuation_reference import estimate_property_scenario
    result=estimate_property_scenario(
        80,1500,"Comparables documentados, fecha 2026-11-13",
        land_area_m2=0,land_pen_m2=0,
    )
    assert result["indicative_total_pen"]==120000
    assert result["status"]=="escenario_documentado_no_pericial"
    assert "no constituye" in result["disclaimer"].lower()
    import pytest
    with pytest.raises(ValueError):
        estimate_property_scenario(80,1500,"no")
    with pytest.raises(ValueError):
        estimate_property_scenario(80,1500,"Comparables documentados",land_area_m2=30)


def test_financial_scenario_endpoint_updates_pdf(monkeypatch):
    monkeypatch.setenv("YOLO_CPU_ENABLED","1")
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    monkeypatch.setattr(app,"predict_cracks",stub_yolo)
    client=client_login()
    created=client.post("/api/quick-inspection",
       files={"exterior":("fachada.jpg",photo(),"image/jpeg")})
    assert created.status_code==201
    cid=created.json()["case_id"]
    response=client.post("/api/cases/"+cid+"/valuation-scenario",json={
        "built_area_m2":80,"construction_unit_pen_m2":1500,
        "source":"Ejemplo documentado de costo por m2, 2026",
    })
    assert response.status_code==200,response.text
    assert response.json()["indicative_total_pen"]==120000
    report=client.get("/api/cases/"+cid+"/quick-report").json()
    assert report["documented_valuation_scenario"]["indicative_total_pen"]==120000
    pdf=client.get("/api/cases/"+cid+"/quick-report.pdf")
    assert pdf.status_code==200 and pdf.content.startswith(b"%PDF")
