"""Tests of YOLOv8 segmentation decoding without downloading model weights."""
from types import SimpleNamespace

import numpy as np
from PIL import Image

import yolo_onnx


class FakeSession:
    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def run(self, outputs, feed):
        assert feed["images"].shape == (1, 3, 640, 640)
        # One anchor, one class, 32 mask coefficients.
        detections = np.zeros((1, 37, 1), dtype=np.float32)
        detections[0, :4, 0] = [320, 320, 400, 160]
        detections[0, 4, 0] = 0.91
        detections[0, 5, 0] = 10
        prototypes = np.ones((1, 32, 160, 160), dtype=np.float32)
        prototypes[:, 1:, :, :] = 0
        return [detections, prototypes]


def test_segmentation_mask_decoded_and_overlay_created(monkeypatch, tmp_path):
    monkeypatch.setattr(yolo_onnx, "get_session", lambda: FakeSession())
    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (120, 60), "white").save(image_path)
    found, overlay = yolo_onnx.predict_cracks(image_path)
    assert overlay.size == (120, 60)
    assert len(found) == 1
    assert found[0]["has_mask"] is True
    assert 0 < found[0]["mask_coverage_percent_image"] < 100
    assert "anchor" not in found[0] and "raw_box" not in found[0]
    assert overlay.getpixel((60, 30)) != (255, 255, 255)
    # Letterboxed image: mask must not paint outside the predicted box.
    assert overlay.getpixel((0, 0)) == (255, 255, 255)


def test_no_mask_prototypes_does_not_fabricate_segmentation(monkeypatch, tmp_path):
    fake = FakeSession()
    original_run = fake.run
    fake.run = lambda outputs, feed: original_run(outputs, feed)[:1]
    monkeypatch.setattr(yolo_onnx, "get_session", lambda: fake)
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (120, 60), "white").save(path)
    found, overlay = yolo_onnx.predict_cracks(path)
    assert len(found) == 1
    assert found[0]["has_mask"] is False
    assert found[0]["mask_coverage_percent_image"] is None
    assert overlay.size == (120, 60)


def test_zero_detections_do_not_produce_false_coverage(monkeypatch, tmp_path):
    fake = FakeSession()
    original = fake.run
    def no_detections(outputs, feed):
        predictions, prototypes = original(outputs, feed)
        predictions[0, 4, 0] = 0.1
        return [predictions, prototypes]
    fake.run = no_detections
    monkeypatch.setattr(yolo_onnx, "get_session", lambda: fake)
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (80, 100), "white").save(path)
    found, _ = yolo_onnx.predict_cracks(path)
    assert found == []
