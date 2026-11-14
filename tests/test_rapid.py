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


def test_optional_paint_detector_integrates_only_when_enabled(monkeypatch):
    monkeypatch.setenv("YOLO_CPU_ENABLED","1")
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    monkeypatch.setenv("YOLO_PAINT_ENABLED","1")
    monkeypatch.setenv("YOLO_PAINT_SHA256","a"*64)
    monkeypatch.setattr(app,"predict_cracks",lambda p:([],Image.new("RGB",(80,80))))
    monkeypatch.setattr(app,"detect_paint",lambda p:([
        {"category":"pintura_deteriorada","confidence":0.87,
         "description":"Posible daño de pintura","box":[5,5,40,30],
         "improvement":"Inspeccionar humedad y repintar"}
    ],Image.new("RGB",(80,80))))
    client=client_login()
    response=client.post("/api/quick-inspection",
        files={"exterior":("fachada.jpg",photo(),"image/jpeg")})
    assert response.status_code==201,response.text
    data=response.json()
    assert "pintura_deteriorada" in [x["category"] for x in data["experimental_issues"]]
    assert "pintura_deteriorada" not in [x["category"] for x in data["issues"]]
    assert all(not x["affects_condition"] and not x["affects_quote"] for x in data["experimental_issues"])
    assert data["image_summaries"][0]["annotated_url"]
    assert "yolo_paint_experimental" in data["models"][0]
    assert data["quote"]["total_pen"] is None
    assert data["property_market_value"] is None
    assert data["human_review_required"]
    assert client.get("/api/cases/"+data["case_id"]+"/quick-report.pdf").status_code==200


def test_official_reference_sierra_figures_and_example():
    from official_reference import reference_scenario, official_unit_values
    values=official_unit_values()
    assert values["zones"]["sierra"]["D"]==1027.37
    example=reference_scenario(60,"sierra","D",illustrative=True)
    assert example["construction_reference_pen"]==61642.20
    assert example["market_value_pen"] is None
    assert example["land_value_pen"] is None
    assert example["area_user_reported"] is False
    assert "SIMULACIÓN" in example["disclaimer"]
    import pytest
    with pytest.raises(ValueError):
        reference_scenario(0,"sierra","D")
    with pytest.raises(ValueError):
        reference_scenario(60,"sierra","Z")


def test_official_construction_reference_api_persists_in_pdf(monkeypatch):
    monkeypatch.setenv("YOLO_CPU_ENABLED","1")
    monkeypatch.setenv("YOLO_DEFECT5_ENABLED","0")
    monkeypatch.setenv("YOLO_PAINT_ENABLED","0")
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    monkeypatch.setattr(app,"predict_cracks",stub_yolo)
    from fastapi.testclient import TestClient
    assert TestClient(app.app).get("/api/official-construction-values").status_code==401
    client=client_login()
    catalog=client.get("/api/official-construction-values")
    assert catalog.status_code==200
    assert catalog.json()["zones"]["sierra"]["D"]==1027.37
    created=client.post("/api/quick-inspection",
        files={"exterior":("foto.jpg",photo(),"image/jpeg")})
    assert created.status_code==201,created.text
    cid=created.json()["case_id"]
    assert created.json()["illustrative_construction_example"]["construction_reference_pen"]==61642.20
    assert created.json()["property_market_value"] is None
    result=client.post("/api/cases/"+cid+"/official-construction-reference",
        json={"built_area_m2":60,"zone":"sierra","grade":"D"})
    assert result.status_code==200,result.text
    assert result.json()["construction_reference_pen"]==61642.20
    assert result.json()["market_value_pen"] is None
    assert result.json()["land_value_pen"] is None
    stored=client.get("/api/cases/"+cid+"/quick-report")
    assert stored.status_code==200
    assert stored.json()["official_construction_reference"]["construction_reference_pen"]==61642.20
    assert stored.json()["property_market_value"] is None
    pdf=client.get("/api/cases/"+cid+"/quick-report.pdf")
    assert pdf.status_code==200
    assert pdf.content.startswith(b"%PDF")
    assert len(pdf.content)>2500
    invalid=client.post("/api/cases/"+cid+"/official-construction-reference",
        json={"built_area_m2":60,"zone":"sierra","grade":"A"})
    assert invalid.status_code==422


def test_automatic_repair_scenario_with_trusted_public_prices():
    from repair_pricing import build_automatic_quote
    issues=[
        {"category":"fisura_visible","shot":"Exterior","confidence":0.74},
        {"category":"fisura_visible","shot":"Exterior","confidence":0.36},
        {"category":"fisura_visible","shot":"Exterior","confidence":0.90},
    ]
    quote=build_automatic_quote(issues)
    assert quote["status"]=="escenario_automatico_ilustrativo"
    assert len(quote["items"])==1
    assert quote["illustrative_min_pen"]==23.02
    assert quote["illustrative_max_pen"]==69.06
    assert quote["items"][0]["unit"]=="m"
    assert quote["items"][0]["source_url"].startswith("https://")
    assert quote["items"][0]["quantity"] is None
    assert quote["total_pen"] is None
    assert quote["suggested_credit_pen"] is None


def test_repair_quote_prefers_highest_confidence_regardless_of_detection_order():
    from repair_pricing import build_automatic_quote
    scenario=build_automatic_quote([
        {"category":"fisura_visible","shot":"Exterior","confidence":0.36},
        {"category":"fisura_visible","shot":"Exterior","confidence":0.74},
        {"category":"fisura_visible","shot":"Exterior","confidence":0.59},
    ])
    assert len(scenario["items"])==1
    assert scenario["items"][0]["confidence"]==0.74
    assert scenario["illustrative_min_pen"]==23.02
    assert scenario["illustrative_max_pen"]==69.06
    assert scenario["low_confidence_excluded"]==0


def test_automatic_repair_excludes_structural_cracks_and_experimental_paint():
    from repair_pricing import build_automatic_quote
    quote=build_automatic_quote([
        {"category":"grieta_escalonada","shot":"Exterior","confidence":0.98},
        {"category":"pintura_deteriorada","shot":"Interior","confidence":0.97},
        {"category":"fisura_visible","shot":"Exterior","confidence":0.36},
    ])
    assert quote["illustrative_min_pen"] is None
    assert quote["illustrative_max_pen"] is None
    assert len(quote["excluded"])==3
    assert quote["total_pen"] is None
    assert quote["low_confidence_excluded"]==1


def test_automatic_repair_combines_distinct_views_and_types():
    from repair_pricing import build_automatic_quote
    quote=build_automatic_quote([
        {"category":"fisura_visible","shot":"Exterior","confidence":0.8},
        {"category":"fisura_visible","shot":"Interior","confidence":0.8},
        {"category":"moho_visible","shot":"Interior","confidence":0.8},
        {"category":"pintura_desprendida","shot":"Interior","confidence":0.8},
    ])
    assert len(quote["items"])==4
    assert quote["illustrative_min_pen"]==163.31
    assert quote["illustrative_max_pen"]==489.93


def test_automatic_quote_available_in_new_and_old_reports_pdf(monkeypatch):
    monkeypatch.setenv("YOLO_CPU_ENABLED","1")
    monkeypatch.setenv("YOLO_DEFECT5_ENABLED","0")
    monkeypatch.setenv("YOLO_PAINT_ENABLED","0")
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    monkeypatch.setattr(app,"predict_cracks",stub_yolo)
    client=client_login()
    created=client.post("/api/quick-inspection",files={
        "exterior":("fachada.jpg",photo(),"image/jpeg")})
    assert created.status_code==201,created.text
    body=created.json()
    scenario=body["automatic_repair_scenario"]
    assert scenario["illustrative_min_pen"]==23.02
    assert scenario["illustrative_max_pen"]==69.06
    assert body["quote"]["total_pen"] is None
    cid=body["case_id"]
    # Legacy reports still compute the automatic illustrative scenario.
    with app.db() as con:
        row=con.execute("SELECT result_json FROM rapid_reports WHERE case_id=?",(cid,)).fetchone()
        stored=__import__("json").loads(row["result_json"])
        stored.pop("automatic_repair_scenario",None)
        con.execute("UPDATE rapid_reports SET result_json=? WHERE case_id=?",
                    (__import__("json").dumps(stored),cid))
    previous=client.get("/api/cases/"+cid+"/quick-report")
    assert previous.status_code==200,previous.text
    assert previous.json()["automatic_repair_scenario"]["illustrative_max_pen"]==69.06
    pdf=client.get("/api/cases/"+cid+"/quick-report.pdf")
    assert pdf.status_code==200 and pdf.content.startswith(b"%PDF")


def test_quick_ai_fills_advanced_case_sections_without_duplicating_or_approving(monkeypatch):
    monkeypatch.setenv("YOLO_CPU_ENABLED","1")
    monkeypatch.setenv("YOLO_DEFECT5_ENABLED","0")
    monkeypatch.setenv("YOLO_PAINT_ENABLED","0")
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    monkeypatch.setattr(app,"predict_cracks",stub_yolo)
    client=client_login()
    created=client.post("/api/quick-inspection",
        files={"exterior":("fachada.jpg",photo(),"image/jpeg")})
    assert created.status_code==201,created.text
    report=created.json()
    cid=report["case_id"]
    case=client.get("/api/cases/"+cid)
    assert case.status_code==200
    data=case.json()
    assert data["quick_report"]["case_id"]==cid
    assert data["quick_report"]["images_analyzed"]==1
    assert data["quick_report"]["automatic_repair_scenario"]["total_pen"] is None
    assert len(data["evidence"])==1
    assert len(data["visual_assessments"])==1
    assert len(data["findings"])==1
    assert data["findings"][0]["origin"]=="quick_ai"
    assert data["findings"][0]["state"]=="pendiente"
    assert data["findings"][0]["severity"]=="no_determinada"
    assert data["findings"][0]["adjustment"]==0
    assert data["base_value"]==0
    assert data["quick_findings_need_sync"] is False
    first=client.post("/api/cases/"+cid+"/sync-quick-report")
    assert first.status_code==200
    assert first.json()["created"]==0
    # Keep the review untouched when synchronization is requested later.
    fid=data["findings"][0]["id"]
    client.put("/api/findings/"+fid+"/review",json={
        "state":"confirmado","adjustment":0,"justification":"Verificado en inspección"})
    resync=client.post("/api/cases/"+cid+"/sync-quick-report")
    assert resync.status_code==200 and resync.json()["created"]==0
    after=client.get("/api/cases/"+cid).json()
    assert len(after["findings"])==1
    assert after["findings"][0]["state"]=="confirmado"
    assert after["estimated_value"]==0


def test_legacy_quick_report_synchronizes_findings_once(monkeypatch):
    monkeypatch.setenv("YOLO_CPU_ENABLED","1")
    monkeypatch.setenv("YOLO_DEFECT5_ENABLED","0")
    monkeypatch.setenv("YOLO_PAINT_ENABLED","0")
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    monkeypatch.setattr(app,"predict_cracks",stub_yolo)
    client=client_login()
    body=client.post("/api/quick-inspection",
        files={"exterior":("fachada.jpg",photo(),"image/jpeg")}).json()
    cid=body["case_id"]
    with app.db() as con:
        con.execute("DELETE FROM findings WHERE case_id=? AND origin='quick_ai'",(cid,))
    before=client.get("/api/cases/"+cid).json()
    assert before["quick_findings_need_sync"] is True
    assert before["quick_report"]["issues"]
    first=client.post("/api/cases/"+cid+"/sync-quick-report")
    assert first.status_code==200 and first.json()["created"]==1
    second=client.post("/api/cases/"+cid+"/sync-quick-report")
    assert second.status_code==200 and second.json()["created"]==0
    after=client.get("/api/cases/"+cid).json()
    assert after["quick_findings_need_sync"] is False
    assert len(after["findings"])==1
