# Tasación Visual Rápida · Tarea 8

Aplicación local con FastAPI, SQLite y dos perfiles de YOLO. **Vivienda usa YOLOv8n-seg de OpenSistemas para segmentar fisuras visibles**. Los otros tipos usan YOLOv8n COCO para detección genérica. Las demás clases de daños y materiales se pueden registrar manualmente y el panel muestra cuáles siguen sin soporte automático.

## Prueba rápida de una fisura

1. Crea un expediente **Vivienda**.
2. En **Capturar evidencia**, adjunta `tests/fisura-superficie.jpg` como **Detalles**.
3. En **Analizar**, selecciona la foto y pulsa **Analizar con YOLO**. En la prueba de integración, la máscara tuvo 529 vértices y el modelo asignó 89,30% de confianza a la detección.
4. Selecciona **fisura visible** para resaltar la región. En **Revisar hallazgos**, confirma o descarta la propuesta y documenta el motivo. El resultado automático queda pendiente y no genera descuentos. Los ajustes de valor requieren respaldo documentado.
5. Si quieres comparar con la prueba genérica del autobús, crea un expediente **Mototaxi** y carga `tests/bus.jpg` como **Frente**. Esta imagen comprueba el modelo genérico.

## Evaluación del perfil de fisuras

Se evaluó el checkpoint en las 112 imágenes y 148 anotaciones del split público de prueba de Crack-Seg, con Ultralytics 8.4.211, CPU, tamaño de inferencia 640 y parámetros de validación predeterminados. Las métricas para **máscaras** fueron precisión 0,734, exhaustividad 0,601, mAP@0,50 0,584 y mAP@0,50–0,95 0,203. Consulta `tests/metricas-crack-seg.json` para los valores completos. Este split incluye superficies de distintos tipos y pertenece al mismo dataset declarado por el autor. Falta medir la generalización con viviendas locales y los falsos positivos en superficies sin daños. La confianza de una detección se refiere a esa región y no determina profundidad, causa, riesgo estructural ni pérdida de valor.

El peso de [OpenSistemas YOLOv8-crack-seg](https://huggingface.co/OpenSistemas/YOLOv8-crack-seg) se conserva en `weights/vivienda-fisuras.pt`. Licencia declarada AGPL-3.0. SHA-256 `cce8b30758621622a03119677709d172f2ec1a2f0452279bda37732fcffb00af`. La imagen de prueba corresponde a `images/test/1806.rf.8c87862452f4ce0232a666c180ec9942.jpg` del [dataset público Crack-Seg](https://docs.ultralytics.com/datasets/segment/crack-seg/), acreditado a University en Roboflow Universe.

## Windows · Python 3.12 de 64 bits

En PowerShell, dentro de la carpeta extraída:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

Abre [http://127.0.0.1:8000](http://127.0.0.1:8000). Usa el usuario **revisor** y la contraseña que muestra la terminal. Para mantener una contraseña entre arranques, configura `$env:TASACION_REVISOR_PASSWORD` en tu sesión antes de ejecutar el script. Cierra la ventana privada del navegador para probar otra cuenta.

Puedes habilitar las cuentas locales **operador** y **administrador** con `$env:TASACION_OPERADOR_PASSWORD` y `$env:TASACION_ADMINISTRADOR_PASSWORD`. Operador registra y captura. Revisor y administrador revisan y cierran. Las cuentas habilitadas comparten el conjunto de expedientes de esta instalación. SQLite guarda imágenes y expedientes en `data/cases.sqlite3`; `TASACION_DB` permite cambiar la ruta. Esta configuración está diseñada para ejecución local en loopback.

## Demostración con YOLO y tasación con evidencia

1. Crea un expediente según el activo que quieras evaluar. Adjunta una imagen a la toma correspondiente.
2. En **Analizar**, selecciona la imagen y pulsa **Analizar con YOLO**. Las clases COCO genéricas quedan fuera de los ajustes. Un revisor puede repetir el análisis y archivar el anterior conservando hallazgos manuales.
3. Para una inspección de vivienda, crea otro expediente y carga sus fotos o videos en las tomas correspondientes. Los avisos de calidad son heurísticos. El video se limita a 60 s, toma hasta diez muestras y conserva los fotogramas que pasan los controles de calidad y similitud. Un video pertenece a la toma elegida y el sistema no infiere el ángulo físico.
4. En **Revisar hallazgos**, confirma las tomas inspeccionadas. Agrega un hallazgo manual dibujando su región o revisa una detección existente. Puedes agrupar dos registros del mismo daño confirmando uno y descartando el otro como duplicado. Para máquina, guarda el resultado de funcionamiento y puedes adjuntar evidencia de video en la toma correspondiente.
5. Ingresa primero el valor base y su fuente. En cada factor, documenta por separado reparación e impacto de mercado. Marca **Ya incluido en el valor base** cuando corresponda. Si cambia la base, vuelve a revisar los ajustes previos. Completa revisión y descarga PDF o JSON.

El puntaje es una regla referencial con pesos iguales por toma confirmada y penalizaciones de 6, 15 y 30 según severidad revisada. La política exige al menos 70% de cobertura para calcular crédito. Un hallazgo severo lo deja pendiente. El crédito de garantía usa el límite inferior del valor y queda pendiente si quedan ajustes sin cuantificar. El crédito de mejora usa presupuesto documentado y porcentaje. La diferencia entre límites del valor es un escenario de ajuste basado en fuentes ingresadas.

## Entrenar pesos especializados

Prepara el dataset con imágenes y etiquetas YOLO en `images/{train,val,test}` y `labels/{train,val,test}`. Copia `datasets/vivienda.example.yaml` y rellena `datasets/manifest.example.csv` con todas las imágenes, asignando cada activo a un solo conjunto. Para segmentación, etiqueta polígonos y pasa `--task segment`.

```powershell
.\.venv\Scripts\python.exe train.py vivienda datasets/vivienda.example.yaml datasets/manifest.csv --epochs 70
```

Se genera `best.pt` y un reporte de evaluación en el conjunto de prueba. Copia el peso a `weights/vivienda.pt` y reemplaza el perfil de vivienda por `"vivienda": "weights/vivienda.pt"` en `specialized` dentro de `config.json`. Reinicia la aplicación. Para probar un peso específico con todos los activos, configura `$env:YOLO_WEIGHTS` antes de iniciar. Instala únicamente pesos de una fuente de confianza.

## Evaluación opcional de ML tabular

Completa el CSV de `datasets/prices.example.csv` con precios verificados, fuentes y al menos 30 activos distintos por categoría. Instala `requirements-ml.txt` y ejecuta `python evaluate_prices.py vivienda ruta/precios.csv`. El script mide error absoluto, sesgo y cobertura empírica del rango en activos separados. Compara con la mediana del entrenamiento. Los resultados son experimentales y el motor de tasación usa evidencia y ajustes revisados.

## Pruebas y origen de los pesos

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Peso [yolov8n.pt de Ultralytics](https://github.com/ultralytics/assets/releases/tag/v8.3.0), licencia AGPL-3.0 incluida en `weights/`. SHA-256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`. Imagen [bus.jpg del repositorio Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/bus.jpg), usada como evidencia de integración.

El ZIP incluye las herramientas y plantillas de entrenamiento y evaluación. La precisión en viviendas, vehículos, maquinaria y terrenos requiere un conjunto real etiquetado y validación por categoría. El valor y el crédito son referenciales para la Tarea 8.

## Visor comparativo y entrenamiento piloto

Activa **Comparar con original** para ver la foto original y la máscara lado a lado. Selecciona un hallazgo desde la lista o desde su región marcada. Puedes ocultar las marcas, aplicar zoom y filtrar por confianza. El resaltado automático después de la inferencia selecciona la detección visible con mayor confianza.

El piloto afinó el checkpoint anterior durante dos épocas con 192 imágenes del split de entrenamiento de Crack-Seg, seleccionadas con semilla 42. Usó CPU, tamaño 416, AdamW y tasa inicial 0,0001. Ambos pesos se evaluaron a tamaño 640 con las 112 imágenes del split público de prueba. Se conservó el modelo anterior como predeterminado, porque obtuvo mayor exhaustividad (60,1% frente a 52,8%) y mAP@0,50 de máscara (58,4% frente a 52,4%). Consulta `experiments/metricas-piloto.json`. Para probar el peso experimental, cambia únicamente la ruta del perfil vivienda en `config.json` a `weights/vivienda-fisuras-piloto.pt` y conserva el alias `crack` a `fisura_visible`.

El piloto evalúa una sola clase. Para incorporar humedad, calamina, ladrillo expuesto y corrosión se requieren imágenes anotadas de esas clases. La validación independiente con viviendas locales sigue pendiente.

## Auditoría de entrega

Consulta `AUDITORIA.md` para el inventario, las correcciones y los pendientes.

## Versión 6 · Comparables y preparación de datos

En **Revisar hallazgos**, registra comparables con referencia, fuente, tipo de precio, precio observado, precio ajustado y justificación. Con dos o más referencias distintas puedes usar el mínimo y el máximo de los precios ajustados como valor base. Los comparables se guardan en el expediente y se incluyen en PDF y JSON. Este rango refleja los comparables ingresados y requiere validación con tasaciones reales. Si cambia la evidencia usada para derivar la base, revisa de nuevo los descuentos.

En **Emitir informe**, el revisor puede exportar anotaciones YOLO de hallazgos confirmados. El ZIP trae cajas para detección, fotos, orden de clases y manifiesto. Antes de entrenar, completa las anotaciones de cada foto, añade negativos y separa train, val y test por activo. Para un modelo de segmentación, se requieren polígonos anotados.

## Acceso desde la pantalla de inicio

Abre la dirección de Uvicorn en tu navegador e introduce el usuario **revisor** y la contraseña que muestra PowerShell. La sesión dura hasta ocho horas y se cierra al reiniciar el servidor. Usa **Cerrar sesión** para cambiar de cuenta. La foto de la portada es un ejemplo incluido y puedes analizarla cargando `tests/fisura-superficie.jpg` en la toma **Detalles**. El flujo se ejecuta en el servidor Python local.
