"""Pipeline engineering tests, using synthetic fixtures ONLY, not training data."""
import json
import sys
from pathlib import Path
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from prepare_housing_dataset import CLASSES, build
from train_missing_categories import validate_dataset_yaml, audit_metrics


def make_dataset(root, missing=None, duplicate=False):
    folder=root/"fixture"
    for split in ("train","val","test"):
        for part in ("images","labels"):(folder/split/part).mkdir(parents=True)
        for cls,label in enumerate(CLASSES):
            if label==missing:continue
            for idx in range(3):
                name=f"{label}_{idx}"
                hue=0 if duplicate else {"train":11,"val":100,"test":189}[split]
                Image.new("RGB",(20,20),(cls*21%255,idx*39+12,hue)).save(folder/split/"images"/(name+".png"))
                (folder/split/"labels"/(name+".txt")).write_text(f"{cls} .5 .5 .25 .25\n")
    config={"sources":[{"id":"fixture","directory":"fixture","license":"CC BY 4.0",
             "license_url":"https://example.org/license",
             "attribution":"Prueba sintética de código, no dataset de viviendas",
             "class_map":{str(i):name for i,name in enumerate(CLASSES)}}]}
    (root/"sources.json").write_text(json.dumps(config))
    return folder


def test_preparation_requires_all_classes(tmp_path):
    make_dataset(tmp_path)
    out=tmp_path/"ready"
    manifest=build(tmp_path/"sources.json",tmp_path,out,minimum=3)
    assert manifest["images"]==81
    assert all(len(manifest["counts"][part])==9 for part in ("train","val","test"))
    assert (out/"data.yaml").is_file()
    validate_dataset_yaml(out/"data.yaml")


def test_missing_class_blocks_training(tmp_path):
    make_dataset(tmp_path,missing="hueco_sin_puerta")
    with pytest.raises(ValueError,match="hueco_sin_puerta"):
        build(tmp_path/"sources.json",tmp_path,tmp_path/"ready",minimum=3)


def test_duplicate_images_blocked(tmp_path):
    make_dataset(tmp_path,duplicate=True)
    with pytest.raises(ValueError,match="Fuga de datos"):
        build(tmp_path/"sources.json",tmp_path,tmp_path/"ready",minimum=3)


def test_unlicensed_source_rejected(tmp_path):
    make_dataset(tmp_path)
    path=tmp_path/"sources.json";cfg=json.loads(path.read_text())
    cfg["sources"][0]["license"]="UNKNOWN";path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match="licencia"):
        build(path,tmp_path,tmp_path/"ready",minimum=3)


def test_missing_ground_truth_label_blocks(tmp_path):
    folder=make_dataset(tmp_path)
    next((folder/"train"/"labels").glob("*.txt")).unlink()
    with pytest.raises(ValueError,match="Falta anotación"):
        build(tmp_path/"sources.json",tmp_path,tmp_path/"ready",minimum=3)


def test_every_class_metric_must_pass():
    class Box:
        ap_class_index=list(range(9))
        ap50=[.75]*9
        r=[.8]*9
        map50=.75
    class Result:box=Box()
    assert audit_metrics(Result())["passed_per_class"]
    Box.ap50[6]=.04
    result=audit_metrics(Result())
    assert not result["passed_per_class"]
    assert result["class_metrics"]["calamina_visible"]["ap50"]==.04
