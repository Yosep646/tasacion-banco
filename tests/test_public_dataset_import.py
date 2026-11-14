"""Test public-dataset importer guards without downloading network photos."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from import_roboflow_source import import_source


def test_import_requires_license_review(tmp_path):
    with pytest.raises(ValueError,match="revisión humana"):
        import_source("buildinghealth","defect-images-property-ssd5a",1,tmp_path/"import",
                      "CC BY 4.0","https://universe.roboflow.com/buildinghealth/defect-images-property-ssd5a",
                      "Dataset source",{"0":"piso_roto"},reviewed=False)


def test_import_requires_key(tmp_path,monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY",raising=False)
    with pytest.raises(RuntimeError,match="ROBOFLOW_API_KEY"):
        import_source("buildinghealth","defect-images-property-ssd5a",1,tmp_path/"import",
                      "CC BY 4.0","https://universe.roboflow.com/buildinghealth/defect-images-property-ssd5a",
                      "Dataset source",{"0":"piso_roto"},reviewed=True)


def test_import_rejects_unlicensed_categories(tmp_path):
    with pytest.raises(ValueError,match="no pertenecen"):
        import_source("buildinghealth","defect-images-property-ssd5a",1,tmp_path/"import",
                      "CC BY 4.0","https://example.org/license",
                      "Source",{"0":"animal"},reviewed=True)
