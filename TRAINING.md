# Entrenamiento de los 9 elementos restantes de viviendas

**Estado real (14/11/2026):** pipeline creado y probado; no existe un dataset completo y autorizado en el repositorio. No hay pesos entrenados y validados para estas nueve clases. La aplicación de Render conserva los modelos de fisuras y cinco defectos que ya estaban verificados.

## Etiquetas requeridas (YOLO detection, en este orden)

| ID | Etiqueta | Cómo anotarla correctamente |
|---|---|---|
| 0 | piso_roto | Baldosa o superficie del piso visiblemente fracturada, no piso simplemente sucio |
| 1 | pared_sin_tarrajeo | Región de pared de obra sin revestimiento, no daño inferido |
| 2 | pared_tarrajeada | Superficie visible que muestra tarrajeo; etiqueta regional verificada |
| 3 | ladrillo_expuesto | Ladrillos reconocibles a simple vista |
| 4 | puerta_visible | Puerta identificable (cerrada o abierta) |
| 5 | hueco_sin_puerta | Vano arquitectónico abierto SIN hoja instalada, confirmado visualmente |
| 6 | calamina_visible | Cubierta de plancha metálica ondulada, donde sea reconocible |
| 7 | techo_concreto_visible | Cubierta de apariencia concreta; NO se puede certificar el material desde una foto |
| 8 | pintura_deteriorada | Descascarillado, ampollas o deterioro superficial, no suciedad ambigua |

**No** deducir `hueco_sin_puerta` de la ausencia de `puerta_visible`. No confundir `piso_roto` con manchas. Los ejemplos negativos necesitan anotación revisada por una persona.

## Datos que se necesitan

El repositorio trae `datasets/sources.json` vacío a propósito: no se deben inventar fotografías ni etiquetas. Cada dataset fuente debe tener autorización comercial compatible, URL de licencia comprobada, titular y versión documentados; imágenes reales de interiores/exteriores (con consentimiento y datos sensibles eliminados), etiquetas YOLO para las clases que realmente represente, y particiones independientes `train/val/test`. El script no permite crear un modelo si falta alguna clase en alguna partición o hay duplicados exactos.

Posibles fuentes públicas para investigación (requieren selección y revisión de licencia, etiquetas y dominio antes de incorporarse):

- [Property_defects, Roboflow](https://universe.roboflow.com/image-uhpot/property_defects) declara dominio público y contiene `floor_damage`, que **no equivale necesariamente** a `piso_roto`; revisar imágenes.
- [Door-and-Window Detection, MEnesAtik](https://github.com/MEnesAtik/Door-And-Window-Detection-Dataset) incluye puertas en fachada, pero **no se confirmó licencia de redistribución/comercial**. No usar hasta autorización.
- [Paint Defects, Hugging Face](https://huggingface.co/datasets/imaadd05/paint-defects-roboflow-main) declara CC BY 4.0 y cajas para defectos de pintura; las etiquetas requieren mapeo humano.
- [MINC-2500](https://doi.org/10.1061/JCCEE5.CPENG-7313) usa etiquetas de **clasificación de materiales**, no cajas YOLO; no se puede incorporar directamente.
- [ADE20K](https://huggingface.co/datasets/pictograph/ade-20k) contiene puertas y superficies pero sus **imágenes no están autorizadas para uso comercial** sin permiso del titular.

No existe aquí un conjunto verificado que cubra simultáneamente todos los materiales, defectos, puertas y vanos. Cada fuente debe adaptarse; no unir datasets con etiquetas incompatibles a ciegas.

## Estructura para cargar datos

Dentro del archivo `housing-dataset.zip` la ruta raíz es:

```text
datasets/raw/fuente_1/
├── train/images/...
├── train/labels/...
├── val/images/...
├── val/labels/...
├── test/images/...
└── test/labels/...
```

En `datasets/sources.json` reemplazar el arreglo `sources` vacío por definiciones verificadas:

```json
{
  "sources": [
    {
      "id": "fuente_1",
      "directory": "datasets/raw/fuente_1",
      "license": "CC BY 4.0",
      "license_url": "https://URL-REAL-DE-LICENCIA",
      "attribution": "Nombre del titular, título del dataset y versión",
      "class_map": {"0": "piso_roto"},
      "splits": {"train": "train", "val": "val", "test": "test"}
    }
  ]
}
```

El ejemplo anterior es **una plantilla, no una fuente descargada ni una licencia verificada**. La combinación final debe cubrir las nueve clases en las tres particiones.

## Comandos de preparación y entrenamiento

Instalar dependencias de entrenamiento (recomendado GPU para entrenamiento completo):

```bash
pip install ultralytics pyyaml pillow pytest
python scripts/prepare_housing_dataset.py --registry datasets/sources.json --base . --output datasets/prepared
python scripts/train_missing_categories.py --dataset datasets/prepared/data.yaml --epochs 80 --imgsz 640 --output runs/housing-n9
```

- El pipeline revisa licencias, mapeo de clases, cajas normalizadas, duplicados exactos, distribución train/val/test y muestra las clases con pocas anotaciones. El control mínimo por defecto es **15 cajas en entrenamiento y 3 en validación y test por clase**, que es un mínimo técnico para bloquear dataset vacío, **no significa que sean suficientes para buena precisión**. Una evaluación fiable requiere muchas más imágenes variadas por clase.
- El entrenamiento parte de `yolo26n.pt`, evalúa en test independiente y genera `evaluation_report.json` con **mAP50 y recall por clase**.
- Bloquea la exportación si cualquier clase cae por debajo de AP50 0.40 o recall 0.50; esos son **umbrales experimentales**, no aprobación para decisiones hipotecarias.
- Si pasan los umbrales, exporta `candidate_model.onnx`, pero **no lo sube a Render**. Falta revisar falsas detecciones, sensibilidad por clase, licencia comercial y pruebas en viviendas reales representativas del Perú.
- GitHub Actions ofrece el workflow manual **Train remaining nine housing classes**: requiere secretos de GitHub Actions `TRAINING_DATA_URL` (URL HTTPS privada o firmada de un archivo ZIP autorizado) y `TRAINING_DATA_SHA256` (hash SHA256 del ZIP), además de `datasets/sources.json` completado. El ZIP debe contener exclusivamente rutas bajo `datasets/raw/`; los datos no se publican en GitHub. Los runners habituales de GitHub son CPU y pueden ser demasiado lentos para entrenar el conjunto completo; una GPU es preferible.
- No se publican fotografías personales en GitHub. Los datos `datasets/raw/`, `datasets/prepared/`, `runs/` están en `.gitignore`. No utilices GitHub Releases públicos para fotografías privadas o expedientes de usuarios.

## Verificación de extremo a extremo

```bash
pytest tests/test_training_pipeline.py -q
```

Estas pruebas usan **imágenes sintéticas generadas para probar el pipeline**, no sirven como dataset de entrenamiento y no justifican afirmar que ya se detectan puertas, tarrajeo o pisos rotos.

## Criterios para activar un nuevo modelo en Render

1. Dataset completo, licencia comprobada y revisión visual del etiquetado.
2. Entrenamiento finalizado y artefactos reproducibles.
3. Evaluación independiente por clase, incluidas métricas sobre viviendas peruanas.
4. Versión ONNX con hash SHA256 fijado, prueba real de CPU, memoria y latencia en Render Free.
5. Integración en el endpoint, fallback, despliegue y prueba con consentimientos y fotografías reales.
6. Revisión humana en tasaciones y préstamos; la IA no aprueba créditos ni determina seguridad estructural.


## Importar fotografías reales desde un proyecto público Roboflow (con licencia revisada)

Se añadió `scripts/import_roboflow_source.py` y el registro `datasets/public_candidates.json`. Se identificaron tres fuentes públicas candidatas; la licencia declarada, etiquetas originales y utilidad se documentan por separado. **No se descargaron ni añadieron fotografías de esas fuentes al repositorio**, porque no hay credenciales de exportación disponibles y las clases deben revisarse antes de incorporarlas.

Ejemplo orientativo tras verificar en Roboflow los IDs reales de categorías y la versión del dataset:

```bash
pip install roboflow pyyaml
# Configura ROBOFLOW_API_KEY únicamente en tu entorno local o secreto de CI.
python scripts/import_roboflow_source.py \
  --workspace buildinghealth \
  --project defect-images-property-ssd5a \
  --version 1 \
  --output datasets/raw/defectos_property \
  --license "CC BY 4.0" \
  --license-url "https://universe.roboflow.com/buildinghealth/defect-images-property-ssd5a" \
  --attribution "Defect Images - Property, buildinghealth (versión revisada)" \
  --class-map '{"0":"piso_roto"}' \
  --reviewed
```

**Importante:** `"0"` en el ejemplo es solo un marcador, **NO** una asignación verificada. Debes leer el `data.yaml` original y comprobar que la clase representa superficies fracturadas, no cualquier desperfecto en el piso. El script rechaza mapeos imposibles y guarda la procedencia de cada fuente; no habilita automáticamente pesos en Render.

Las imágenes reales descargadas quedan bajo `datasets/raw/`, fuera de Git. Si no hay credencial, el importador termina con un error explícito en vez de crear ejemplos inventados. Tu consentimiento para usar tus propias fotografías no otorga permisos para usar imágenes privadas de otras personas; para esas fuentes se utiliza la licencia declarada por los titulares.

El otro candidato `paint-defects-roboflow-main` está en Hugging Face, declara CC BY 4.0 y distribuye imágenes con anotaciones COCO. No puede mezclarse con etiquetas YOLO hasta convertir y validar cada caja.
