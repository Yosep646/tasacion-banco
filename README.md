# Tasación Visual — aplicación web referencial

Sistema web en español para registrar expedientes de bienes, capturar fotografías, revisar hallazgos y documentar una **tasación referencial**. Implementado en Python (FastAPI), SQLite y HTML/JavaScript. No sustituye un peritaje profesional ni determina aprobaciones de crédito.

## Estado real de implementación

- ✅ Inicio de sesión por roles: administrador, revisor, operador.
- ✅ Expedientes: vivienda, mototaxi, vehículo, maquinaria y terreno.
- ✅ Evidencias fotográficas JPG, PNG y WebP, con validación, redimensionado y almacenamiento privado.
- ✅ Hallazgos manuales; revisión humana, confirmación y descarte; ajustes documentados.
- ✅ Valor base, fuente y cálculo referencial (valor base menos ajustes confirmados).
- ✅ Exportación PDF y JSON, auditoría por expediente, pruebas automatizadas en GitHub Actions.
- ✅ Endpoint de salud y configuración para Render.
- ✅ Adaptador de segmentación de fisuras YOLOv8n integrado y probado en GitHub Actions con pesos reales, descarga con SHA-256 e inferencia sobre imagen sintética. **El peso no se almacena en GitHub**; se descarga de su fuente pública verificando la suma SHA-256. Para utilizarlo localmente o en Render se debe instalar la dependencia opcional, descargar el modelo y establecer YOLO_VIVIENDA_WEIGHTS. Ninguna detección genera descuentos sin revisión.
- ⚠️ Esta reconstrucción **no incluye** entrenamiento validado, segmentación con máscaras, inspección de video, comparables automáticos, cálculo bancario de crédito ni validación en campo. El README anterior describía capacidades de una entrega ZIP que no está presente en GitHub.

## Ejecutar localmente

Requiere Python 3.12 (o compatible).

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Configura una contraseña de administrador y una clave de sesión:

```bash
# macOS / Linux
export TASACION_ADMIN_PASSWORD='cambia-esta-clave-por-una-segura'
export SESSION_SECRET='clave-aleatoria-larga-de-al-menos-32-caracteres'
uvicorn app:app --reload
```

En PowerShell:

```powershell
$env:TASACION_ADMIN_PASSWORD = 'cambia-esta-clave-por-una-segura'
$env:SESSION_SECRET = 'clave-aleatoria-larga-de-al-menos-32-caracteres'
uvicorn app:app --reload
```

Abre http://127.0.0.1:8000, usuario **administrador** y la contraseña definida. Las cuentas opcionales `revisor` y `operador` se habilitan al iniciar con `TASACION_REVISOR_PASSWORD` y `TASACION_OPERADOR_PASSWORD`. Las contraseñas se configuran **solo al crear cada cuenta en la base**; cambiar la variable después no actualiza automáticamente una cuenta existente.

SQLite y fotografías se guardan en `data/` de forma predeterminada. Configura `TASACION_DATA` y, si procede, `TASACION_DB` para otra ruta. **Evita publicar expedientes reales con datos privados sin revisar el tratamiento de datos, permisos y copias de seguridad.**

## YOLO opcional

```bash
pip install -r requirements-yolo.txt
export YOLO_WEIGHTS=/ruta/absoluta/modelo.pt
```

Reinicia la app. Usa un modelo previamente entrenado y verificado; los pesos no se descargan automáticamente. Si no se configura, el botón de análisis informa que el servicio no está disponible; no simula detecciones. Los modelos genéricos COCO detectan objetos, **no certifican grietas ni daños**.

## Despliegue en Render

Se incluye `render.yaml` con un servicio Python, disco persistente en `/var/data`, variables `APP_ENV=production`, `TASACION_DATA=/var/data`, `SESSION_SECRET` generado y `TASACION_ADMIN_PASSWORD` manual. **El plan starter y el disco persistente pueden generar cargos.** No despliegues en un plan sin almacenamiento persistente si necesitas conservar las imágenes y los casos.

1. Conecta este repositorio en Render y crea el servicio desde Blueprint (`render.yaml`).
2. Define `TASACION_ADMIN_PASSWORD` como secreto antes del primer arranque; no lo pongas en GitHub.
3. Revisa la factura y la configuración del disco, ejecuta el despliegue y verifica `/health` y el flujo de sesión/expediente.
4. Para imágenes y modelo YOLO en producción, instala dependencias opcionales y proporciona un peso validado sin exponer secretos.

**Estado:** configuración preparada, pero este README no afirma que exista un servicio público operativo; debe verificarse en el dashboard de Render.

## Pruebas

```bash
pip install pytest httpx
python -m pytest -q
```

Las pruebas cubren autenticación, creación de expedientes, revisión, valoraciones, exportación PDF/JSON, fotografía, estado sin YOLO y bootstrap del administrador en producción.

## Licencia y modelos

No se incluyen pesos de terceros en este repositorio. El modelo de OpenSistemas tiene licencia AGPL-3.0 y se distribuye como checkpoint .pt (pickle), cuyo formato puede ejecutar código durante la carga. El checksum verifica la integridad de una versión concreta, pero no certifica que el contenido sea seguro. Revisa licencias de modelos y datasets antes de distribuir o usar comercialmente.

## Inferencia especializada verificada en CI

La ejecución [YOLO real inference smoke test](https://github.com/Yosep646/tasacion-banco/actions/workflows/yolo-smoke.yml) instala dependencias CPU, descarga el peso YOLOv8n-seg de OpenSistemas en `weights/vivienda-fisuras.pt`, comprueba SHA-256 `cce8b30758621622a03119677709d172f2ec1a2f0452279bda37732fcffb00af` y ejecuta inferencia sobre una imagen sintética. Primera ejecución corregida: 1 detección; workflow exitoso. **Esto es una prueba funcional, no una evaluación clínica, estructural ni de precisión del detector**. La evaluación de precisión requiere fotos representativas de viviendas reales con etiquetas de referencia.

Configuración local:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements-yolo.txt
python scripts/download_fisuras.py
# macOS/Linux
export YOLO_VIVIENDA_WEIGHTS=weights/vivienda-fisuras.pt
# PowerShell: $env:YOLO_VIVIENDA_WEIGHTS = 'weights/vivienda-fisuras.pt'
uvicorn app:app --reload
```

`render-yolo.yaml` contiene el despliegue con modelo especializado, plan Standard (2 GB RAM) y disco persistente. **Implica cargos recurrentes**. Debe aprobarse en el workspace correcto, configurando una contraseña administrativa y verificando el acceso público. `render.yaml` conserva el despliegue sin instalar el modelo especializado. Ningún archivo YAML, por sí solo, significa que el servidor esté desplegado.
