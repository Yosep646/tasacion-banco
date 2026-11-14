"""Safely import one Roboflow Universe YOLO dataset with explicit provenance.

Only imports sources that a human reviewed for intended labels and license.
Requires ROBOFLOW_API_KEY. Images/labels are saved locally under datasets/raw/
and never committed to git. A dataset is NOT automatically training-ready.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import hashlib

VALID={"CC BY 4.0","CC0 1.0","CC BY-SA 4.0","Apache-2.0","MIT"}

def import_source(workspace,project,version,outdir,license_name,license_url,attribution,
                  mapping,reviewed=False):
    if not reviewed:
        raise ValueError("La licencia y el mapeo requieren revisión humana")
    if license_name not in VALID or not license_url.startswith("https://") or not attribution:
        raise ValueError("Licencia y atribución verificable obligatorias")
    if not mapping or any((not str(k).isdigit()) for k in mapping):
        raise ValueError("Se requiere mapeo de clases por identificador")
    from prepare_housing_dataset import CLASSES
    if any(label not in CLASSES for label in mapping.values()):
        raise ValueError("Una o más etiquetas no pertenecen al reto")
    key=os.getenv("ROBOFLOW_API_KEY","")
    if not key:
        raise RuntimeError("Configura ROBOFLOW_API_KEY en entorno seguro para descargar")
    from roboflow import Roboflow
    rf=Roboflow(api_key=key)
    with tempfile.TemporaryDirectory() as tmp:
        project_ref=rf.workspace(workspace).project(project)
        # Only download public project versions that the user has access to.
        dataset=project_ref.version(version).download("yolov8",location=str(Path(tmp)/"source"))
        base=Path(dataset.location)
        data_config=base/"data.yaml"
        if not data_config.is_file():
            raise ValueError("Falta data.yaml original")
        import yaml
        data=yaml.safe_load(data_config.read_text(encoding="utf-8"))
        names=data.get("names",[])
        ordered=[names[i] if i in names else names[str(i)] for i in range(len(names))] if isinstance(names,dict) else list(names)
        if any(int(i)>=len(ordered) for i in mapping):
            raise ValueError("El mapeo usa clases fuera del conjunto descargado")
        destination=Path(outdir)
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("Directorio destino no vacío; evita mezclas sin auditar")
        splits={"train":data.get("train","train/images"),
                "val":data.get("val","valid/images"),
                "test":data.get("test","test/images")}
        counts={name:0 for name in mapping.values()}
        saved=0
        for split,relative in splits.items():
            img_dir=(base/relative).resolve()
            if not img_dir.is_relative_to(base.resolve()) or not img_dir.is_dir():
                raise ValueError("Partición incompleta: "+split)
            for img in img_dir.iterdir():
                if not img.is_file() or img.suffix.lower() not in (".jpg",".jpeg",".png",".webp"):
                    continue
                label=img_dir.parent/"labels"/(img.stem+".txt")
                if not label.is_file():
                    continue
                # Strictly require class names in source to match the reviewed mapping.
                mapped=[]
                for row in label.read_text(encoding="utf-8").splitlines():
                    parts=row.split()
                    if len(parts)!=5:raise ValueError("Anotación YOLO incompatible")
                    if parts[0] not in mapping:
                        continue
                    try:
                        box=list(map(float,parts[1:]))
                    except ValueError as exc:
                        raise ValueError("Caja inválida") from exc
                    if any(not 0<=v<=1 for v in box) or min(box[2:])<=0:
                        raise ValueError("Caja fuera de [0,1]")
                    mapped.append(parts[0]+" "+" ".join(parts[1:]))
                    counts[mapping[parts[0]]]+=1
                # Do not treat images from a partially mapped dataset as negatives.
                if not mapped:
                    continue
                dst_images=destination/split/"images"
                dst_labels=destination/split/"labels"
                dst_images.mkdir(parents=True,exist_ok=True)
                dst_labels.mkdir(parents=True,exist_ok=True)
                digest=hashlib.sha256(img.read_bytes()).hexdigest()[:16]
                target=img.stem+"_"+digest
                shutil.copy2(img,dst_images/(target+img.suffix.lower()))
                (dst_labels/(target+".txt")).write_text("\n".join(mapped)+"\n",encoding="utf-8")
                saved+=1
        if saved<1:
            raise ValueError("No se encontraron imágenes positivas de las clases mapeadas")
        manifest={
            "provider":"Roboflow Universe","workspace":workspace,"project":project,
            "version":version,"license":license_name,"license_url":license_url,
            "attribution":attribution,"class_map":mapping,
            "original_class_names":ordered,"images_with_mapped_boxes":saved,
            "annotations_per_mapped_class":counts,
            "status":"candidate_requires_visual_annotation_audit",
            "note":"Solo se importaron imágenes positivas de clases explícitamente mapeadas; ninguna anotación demuestra por sí sola precisión del modelo.",
        }
        (destination/"source_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
        return manifest

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workspace",required=True)
    ap.add_argument("--project",required=True)
    ap.add_argument("--version",type=int,required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--license",required=True)
    ap.add_argument("--license-url",required=True)
    ap.add_argument("--attribution",required=True)
    ap.add_argument("--class-map",required=True,help='JSON e.g. {"1":"piso_roto"}')
    ap.add_argument("--reviewed",action="store_true",help="Confirma que verificaste licencia y la equivalencia semántica de clases")
    args=ap.parse_args()
    try:
        result=import_source(args.workspace,args.project,args.version,args.output,
            args.license,args.license_url,args.attribution,json.loads(args.class_map),args.reviewed)
        print(json.dumps(result,indent=2,ensure_ascii=False))
    except (ValueError,RuntimeError,KeyError) as e:
        raise SystemExit("IMPORT BLOCKED: "+str(e)) from e

if __name__=="__main__":main()
