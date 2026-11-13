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
- ⚠️ Inferencia YOLO es **opcional**. Solo funciona si se instalan las dependencias especializadas y se configura un archivo de pesos confiable. **No hay pesos entrenados ni métricas del modelo incluidos en este repositorio.** Las detecciones nunca descuentan valor sin revisión.
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

No se incluyen pesos de terceros en este repositorio. Revisa licencias de modelos y datasets antes de entrenar, distribuir o ejecutar aplicaciones comerciales.
