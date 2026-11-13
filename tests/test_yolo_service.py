"""Unit tests that run without torch or external weight downloads."""
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from yolo_service import get_model_configuration, parse_results, predict_findings


def test_profile_selection(monkeypatch):
    monkeypatch.delenv("YOLO_WEIGHTS", raising=False)
    monkeypatch.delenv("YOLO_VIVIENDA_WEIGHTS", raising=False)
    assert get_model_configuration("vivienda") is None
    monkeypatch.setenv("YOLO_WEIGHTS", "/tmp/coco.pt")
    monkeypatch.setenv("YOLO_VIVIENDA_WEIGHTS", "/tmp/cracks.pt")
    assert get_model_configuration("vivienda")["profile"] == "vivienda_fisuras"
    assert get_model_configuration("maquinaria")["profile"] == "generico"


def test_segmentation_parser():
    box = SimpleNamespace(cls=[0], conf=[0.893])
    mask = SimpleNamespace(xyn=[[(0.1, 0.1), (0.4, 0.4), (0.2, 0.5)]])
    prediction = SimpleNamespace(boxes=[box], masks=mask, names={0: "crack"})
    detection = parse_results([prediction], specialized=True)[0]
    assert detection["category"] == "fisura_visible"
    assert detection["has_mask"] is True
    assert detection["origin"] == "yolo_fisuras"
    assert detection["confidence"] == 0.893


def test_generic_parser_not_mislabeled_as_damage():
    prediction = SimpleNamespace(boxes=[SimpleNamespace(cls=[0], conf=[0.75])],
                                 masks=None, names={0: "bus"})
    detection = parse_results([prediction], specialized=False)[0]
    assert detection["category"] == "objeto_bus"
    assert detection["origin"] == "yolo_generico"


def test_missing_weights_fails_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict_findings(str(tmp_path / "photo.jpg"), {
            "path": str(tmp_path / "missing.pt"), "specialized": True
        })
