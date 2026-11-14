"""Integration contract for optional multimodal vision (no external call)."""
import json
from io import BytesIO

from PIL import Image
import vision_service


def test_multimodal_adapter_posts_image_and_validates_json(monkeypatch, tmp_path):
    picture = tmp_path / "room.jpg"
    Image.new("RGB", (100, 80), "white").save(picture, format="JPEG")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-only")
    monkeypatch.setenv("VISION_MODEL", "gpt-4.1-mini")

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, *args): return b""

    def mocked_urlopen(request, timeout):
        assert request.full_url == "https://api.openai.com/v1/responses"
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["model"] == "gpt-4.1-mini"
        content = payload["input"][0]["content"]
        assert content[0]["type"] == "input_text"
        assert content[1]["type"] == "input_image"
        assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
        result = {"output": [{"content": [{"type": "output_text", "text":
            json.dumps({"condition": "requiere_mejoras", "summary": "Posibles daños visibles",
                        "materials": [{"name": "Concreto", "location": "Pared"}],
                        "issues": [{"category": "fisura_visible", "severity": "leve",
                                    "description": "Fisura posible", "improvement": "Inspección"}],
                        "limitations": ["No se puede ver la estructura"]})
        }]}]}
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, *args): return json.dumps(result).encode("utf-8")
        return Response()

    monkeypatch.setattr(vision_service.urllib.request, "urlopen", mocked_urlopen)
    result = vision_service.analyze_photo(picture, "Interior", "vivienda")
    assert result["condition"] == "requiere_mejoras"
    assert result["materials"][0]["name"] == "Concreto"
    assert result["issues"][0]["category"] == "fisura_visible"
    assert len(result["limitations"]) == 1


def test_empty_or_unconfigured_vision_does_not_fake_result(monkeypatch, tmp_path):
    picture = tmp_path / "room.jpg"
    Image.new("RGB", (100, 80), "white").save(picture, format="JPEG")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    try:
        vision_service.analyze_photo(picture, "Interior", "vivienda")
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("Vision must not fabricate a result")
