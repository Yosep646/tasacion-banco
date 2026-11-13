# Tasación Visual Rápida — Asistente para microcrédito hipotecario o prendario

> **Objetivo académico:** ayudar a documentar el estado visible de viviendas, mototaxis, maquinaria, vehículos y terrenos mediante fotografías guiadas y facilitar una evaluación referencial de mejoras. No emite certificaciones estructurales ni aprueba créditos.

## Flujo del reto

1. Crear expediente y elegir el activo.
2. **Inspección visual IA**: cargar una secuencia guiada de cuatro fotos. Para vivienda: **Fachada, Paredes, Techo, Interior**. Puede fotografiarse desde el celular.
3. Análisis multimodal opcional (si se configura `OPENAI_API_KEY`): devuelve **materiales visibles, problemas observables, condición preliminar, mejoras sugeridas y limitaciones**, mediante JSON validado.
4. Las observaciones quedan **pendientes** hasta su revisión por un administrador o revisor. El modelo no verifica estructura, humedad oculta, cimentación ni instalaciones internas.
5. Solo con las cuatro vistas completas, revisadas y determinables se muestra un **score visual ilustrativo**: 100 menos 5 puntos por observación leve, 12 por moderada, 25 por severa. **No es un score crediticio entrenado ni calibrado**.
6. El revisor documenta valor base y fuente de mercado, y presupuesto de mejoras y su fuente. Puede ejecutar una **simulación académica de microcrédito**: menor de 70% del presupuesto o 30% del valor base. Ninguna política bancaria real ha sido verificada; los límites son parámetros ilustrativos. No hay desembolso, evaluación de solvencia ni decisión de riesgo.
7. Exporta PDF/JSON con hallazgos revisados y sus limitaciones.

## Estado y límites

- ✅ Backend FastAPI y SQLite, autenticación por roles, evidencia fotográfica, reportes PDF/JSON.
- ✅ Flujo de cuatro tomas y análisis multimodal implementado con API del proveedor (opt-in).
- ✅ Resultados de visión marcados pendientes y revisión humana, score transparente condicionado a cobertura.
- ✅ Simulación referencial con presupuesto/valor base y prueba de extremo a extremo usando **modelo simulado solamente en tests**.
- ✅ YOLOv8 especializado de fisuras probado con pesos reales en CI usando imagen sintética.
- ⚠️ **La cuenta de Render Free NO tiene configurado un modelo multimodal por defecto.** Sin `OPENAI_API_KEY`, los botones de análisis explican el motivo y **no fingen detectar materiales o daños**.
- ⚠️ La detección multimodal real no se ha verificado en vivo en Render porque no hay clave accesible en este entorno. La implementación de la API está probada por sustitución controlada durante las pruebas automatizadas, no por una llamada real al servicio.
- ⚠️ Se implementó la secuencia de fotografías guiadas (no videos). Validación local sobre viviendas peruanas, métricas por categoría, evaluación humana y criterios bancarios reales siguen pendientes.
- ⚠️ Render Free no tiene almacenamiento persistente para la base SQLite ni las fotografías. Para producción, usa almacenamiento persistente externo y una base de datos gestionada con backups, consentimiento y política de retención.

## Activar visión multimodal

En la configuración Environment del servicio Render agrega, sin publicarlas en GitHub:

- `OPENAI_API_KEY`: clave API del proveedor con acceso a modelos de visión (puede generar cargos de uso).
- `VISION_MODEL`: opcional, por defecto `gpt-4.1-mini`, sujeto a disponibilidad y facturación.

En el panel **Inspección visual IA**, el usuario verá si está configurado el análisis automático. Antes de enviarlas al proveedor externo, el responsable debe tener autorización para procesar esas imágenes y evitar información personal innecesaria. La entrada es JPEG redimensionada y protegida por sesión; los resultados requieren confirmación.

No hace falta instalar PyTorch en Render Free para este método: la inferencia se ejecuta en el proveedor externo. Para inferencia YOLO local se mantienen `requirements-yolo.txt`, `YOLO_VIVIENDA_WEIGHTS` y las herramientas de descarga/verificación SHA-256.

## Ejecutar localmente

Python 3.12 es el entorno recomendado.

```bash
python -m venv .venv
# Linux/Mac: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Define `TASACION_ADMIN_PASSWORD` (solo en variables de entorno; no en GitHub), `SESSION_SECRET` aleatorio de 32+ caracteres y opcionalmente `OPENAI_API_KEY`. Arranca:

```bash
uvicorn app:app --reload
```

La aplicación abre en `http://127.0.0.1:8000`. Usuario `administrador`; contraseña: la configurada en la variable `TASACION_ADMIN_PASSWORD`. `APP_ENV=production` exige `TASACION_ADMIN_PASSWORD` y `SESSION_SECRET`. Si el usuario ya existe en SQLite, cambiar la variable no cambia su hash.

## Pruebas

```bash
pip install pytest httpx
python -m pytest -q
python tests/check_frontend.py
```

Pruebas de carga de imágenes, roles, valoraciones, PDF, API visual simulada, revisión humana, cobertura y simulación. El workflow especializado también descarga el modelo YOLO con hash conocido y ejecuta inferencia en una imagen sintética.

## Render

El servicio gratuito existente usa `requirements.txt` y se despliega al actualizar la rama `main` de `Yosep646/tasacion-banco`. La URL es `https://tasacion-banco.onrender.com`. El backend puede publicarse sin modelos pesados, pero el análisis multimodal requiere clave API y recursos externos. El archivo `render-yolo.yaml` es una propuesta separada, de pago, que **no** debe confundirse con el servicio gratuito actualmente utilizado.

## Principios de seguridad

Todos los resultados de IA son observaciones preliminares. No asumir que ausencia de fisuras visibles indica seguridad. Para tomar decisiones crediticias hacen falta peritaje cuando corresponda, evidencias verificadas, identificación y valoración de garantías, evaluación de capacidad de pago, consentimiento y revisión de un responsable humano.


## YOLO real en Render Free: detección automática de fisuras

Se implementó un modelo de segmentación **YOLOv8n especializado en fisuras** (OpenSistemas), exportado a ONNX para evitar PyTorch en la instancia Free. El repositorio dispone de:

- `yolo_onnx.py`: motor CPU ONNX Runtime, descarga de modelo de Release con SHA-256, detección de posibles fisuras, supresión de cajas repetidas y anotación de regiones sobre la fotografía.
- `POST /api/cases/{case_id}/automatic-inspection/{evidence_id}`: corre YOLO en la foto y guarda el resultado como propuesta pendiente de revisión.
- `GET /api/evidence/{evidence_id}/yolo-overlay`: devuelve la imagen con regiones señaladas, protegida por autenticación.
- En **Inspección visual**, carga una foto y pulsa **Detectar fisuras con YOLO**; el botón está habilitado si `YOLO_CPU_ENABLED=1`. Cuando no hay grietas detectadas el estado es **no determinable**, jamás "vivienda segura".
- GitHub Actions: `publish-onnx.yml` exporta el modelo original con su checksum y lo publica en el Release `v0.1-crack-onnx`; `onnx-cpu.yml` descarga el modelo y valida inferencia real en CPU, incluida al menos una detección en una imagen sintética. En producción, el modelo se descarga al primer análisis y se almacena en `/tmp/tasacion-models`.

**Límite esencial:** el modelo ONNX detecta *posibles fisuras*, no valida seguridad estructural ni reconoce automáticamente ladrillo, techo de calamina, concreto o tarrajeo. Para materiales y deterioros adicionales se necesitan modelos entrenados y validados con imágenes representativas, o conectar visión multimodal. La revisión manual no sustituye una prueba de IA, y un score de un único tipo de anomalía no es una tasación pericial. No usar las predicciones para aprobar préstamos.

Este backend usa `onnxruntime` (CPU) y `numpy` en `requirements.txt`, sin requerir cuenta de API para el detector especializado. Render Free tiene recursos limitados: la primera inferencia puede tardar, y las descargas se repiten tras reinicios del contenedor al no tener disco persistente. Revisa la licencia **AGPL-3.0** del modelo.
