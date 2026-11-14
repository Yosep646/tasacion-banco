"""Prepare independently split, licensed YOLO datasets for nine housing categories.

sources.json format: {"sources":[{"id":"example","directory":"datasets/raw/example",
"license":"CC BY 4.0","license_url":"https://...","attribution":"Publisher/version",
"class_map":{"0":"puerta_visible"},"splits":{"train":"train","val":"valid","test":"test"}}]}
Each source has split/images and split/labels with YOLO txt annotation files.
Never infer a negative class merely because there is no detection.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, shutil

CLASSES=("piso_roto","pared_sin_tarrajeo","pared_tarrajeada","ladrillo_expuesto",
         "puerta_visible","hueco_sin_puerta","calamina_visible",
         "techo_concreto_visible","pintura_deteriorada")
LICENSES={"CC BY 4.0","CC0 1.0","CC BY-SA 4.0","Apache-2.0","MIT"}
EXT={".jpg",".jpeg",".png",".webp"}

def load_registry(path,base):
    sources=json.loads(path.read_text(encoding="utf-8")).get("sources")
    if not isinstance(sources,list) or not sources:
        raise ValueError("No hay datasets de imágenes reales anotadas")
    ids=set()
    for source in sources:
        key=source.get("id","")
        if not key.isidentifier() or key in ids:raise ValueError("ID inválido o duplicado")
        ids.add(key)
        if source.get("license") not in LICENSES or not str(source.get("license_url","")).startswith("https://") or not source.get("attribution"):
            raise ValueError("Falta licencia o atribución verificable: "+key)
        folder=(base/source.get("directory","")).resolve()
        if not folder.is_relative_to(base.resolve()) or not folder.is_dir():
            raise ValueError("Directorio de imágenes inexistente o fuera de base: "+key)
        source["_folder"]=folder
        mapping=source.get("class_map")
        if not isinstance(mapping,dict) or not mapping:raise ValueError("Falta mapeo explícito: "+key)
        for class_id,target in mapping.items():
            if not str(class_id).isdigit() or target not in CLASSES:
                raise ValueError("Correspondencia de clase inválida en "+key)
        for split in ("train","val","test"):
            name=source.get("splits",{}).get(split,split)
            if not (folder/name/"images").is_dir():raise ValueError("Falta partición "+split+" de "+key)
    return sources

def read_label(path,mapping):
    if not path.is_file():raise ValueError("Falta anotación "+str(path))
    lines=[]
    for lineno,row in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not row.strip():continue
        values=row.split()
        if len(values)!=5 or values[0] not in mapping:
            raise ValueError("Etiqueta no YOLO o sin mapeo "+str(path)+":"+str(lineno))
        try:coords=list(map(float,values[1:]))
        except ValueError as exc:raise ValueError("Coordenadas inválidas") from exc
        if not all(0<=x<=1 for x in coords) or min(coords[2:])<=0:
            raise ValueError("Caja fuera de límites "+str(path))
        lines.append(str(CLASSES.index(mapping[values[0]]))+" "+" ".join(f"{x:.8f}" for x in coords))
    return lines

def build(registry_path,base,target,minimum=15):
    sources=load_registry(registry_path,base)
    if target.exists() and any(target.iterdir()):raise ValueError("Destino no vacío; no sobrescribir")
    counts={x:Counter() for x in ("train","val","test")}
    duplicates=set()
    entries=[]
    provenance=[]
    for source in sources:
        for split in ("train","val","test"):
            name=source.get("splits",{}).get(split,split)
            image_dir=source["_folder"]/name/"images"
            for image in sorted(image_dir.iterdir()):
                if not image.is_file() or image.suffix.lower() not in EXT:continue
                annotations=read_label(source["_folder"]/name/"labels"/(image.stem+".txt"),source["class_map"])
                if not annotations and not source.get("reviewed_negative_images",False):
                    raise ValueError("Una imagen sin cajas necesita revisión explícita: "+str(image))
                digest=hashlib.sha256(image.read_bytes()).hexdigest()
                if digest in duplicates:raise ValueError("Fuga de datos: imagen idéntica repetida entre splits: "+str(image))
                duplicates.add(digest)
                for ann in annotations:counts[split][CLASSES[int(ann.split()[0])]]+=1
                entries.append((split,source["id"],image,annotations,digest))
        provenance.append({k:source[k] for k in ("id","license","license_url","attribution","class_map")})
    if not entries:raise ValueError("No hay imágenes")
    for split in ("train","val","test"):
        for label in CLASSES:
            required=minimum if split=="train" else max(3,minimum//5)
            if counts[split][label]<required:
                raise ValueError(f"No entrenable: {label} en {split}: {counts[split][label]} cajas, mínimo {required}")
    target.mkdir(parents=True,exist_ok=True)
    try:
        for split,source,image,annotations,digest in entries:
            image_dir=target/split/"images";label_dir=target/split/"labels"
            image_dir.mkdir(parents=True,exist_ok=True);label_dir.mkdir(parents=True,exist_ok=True)
            filename=f"{source}_{digest[:16]}{image.suffix.lower()}"
            shutil.copy2(image,image_dir/filename)
            (label_dir/(Path(filename).stem+".txt")).write_text("\n".join(annotations)+"\n",encoding="utf-8")
        yaml="path: "+str(target.resolve())+"\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n"
        yaml+="".join(f"  {i}: {label}\n" for i,label in enumerate(CLASSES))
        (target/"data.yaml").write_text(yaml,encoding="utf-8")
        report={"classes":list(CLASSES),"images":len(entries),"counts":{k:dict(v) for k,v in counts.items()},
                "sources":provenance,"split_deduplicated":True,
                "labels_require_human_review":True,"status":"dataset_prepared_not_trained"}
        (target/"dataset_manifest.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
        return report
    except Exception:
        shutil.rmtree(target)
        raise

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--registry",required=True,type=Path)
    p.add_argument("--base",type=Path,default=Path("."))
    p.add_argument("--output",type=Path,default=Path("datasets/prepared"))
    p.add_argument("--min-train-boxes",type=int,default=15)
    a=p.parse_args()
    if a.min_train_boxes<1:raise SystemExit("El mínimo debe ser positivo")
    try:print(json.dumps(build(a.registry,a.base.resolve(),a.output,a.min_train_boxes),indent=2,ensure_ascii=False))
    except (OSError,ValueError) as exc:raise SystemExit("DATASET BLOCKED: "+str(exc)) from exc

if __name__=="__main__":main()
