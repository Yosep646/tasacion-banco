## Cotización automática ilustrativa de reparaciones (14/11/2026)

La evaluación rápida ahora devuelve dos resultados económicos **independientes**:
- **Rango automático ILUSTRATIVO de reparaciones:** el motor `repair_pricing.py` toma categorías YOLO con confianza de al menos 50 %, deduplica por tipo de daño y vista, compara con tarifas públicas, y presenta un ejemplo de **1 a 3 m o m² por partida**, SIN conocer la superficie real. El resultado es un escenario de costo para orientar la inspección, **no un presupuesto real, valor comercial, costo final ni aprobación de crédito**. No suma detecciones repetidas de la misma clase/vista; los trabajos de distintas clases pueden solaparse.
- **Referencia de edificación:** calculada con área declarada, zona y categoría de las tablas oficiales CAP 2026; **no incluye terreno ni depreciación**.

Tarifas documentadas para ejemplos de reparaciones:
- Sellado superficial de mortero exterior (CYPE RYY022), **S/23,02/m**: https://peru.generadordeprecios.info/rehabilitacion/Revestimientos/RY_Preparaciones__protecciones__r/Reparaciones/RYY022_Reparacion_de_grieta_en_revestimien.html
- Sellado superficial interior (CYPE RYY022), **S/22,29/m**: https://peru.generadordeprecios.info/rehabilitacion/Revestimientos/RY_Preparaciones__protecciones__r/Reparaciones/RYY022_Reparacion_de_grieta_en_revestimien_0_0_0_0_0_0_0_0_0.html
- Tratamiento de humedad **solo si se confirma capilaridad** (CYPE NHM010), **S/102,88/m²**: https://peru.generadordeprecios.info/rehabilitacion/Aislamientos_e_impermeabilizaciones/NH_Tratamiento_de_humedades_por_c/Morteros/NHM010_Tratamiento_de_humedades_por_capila.html
- Mano de obra de tratamiento superficial de moho, **S/44/m²**, y pintura desprendida, hipótesis raspado **S/13/m²** + pintura estándar con material **S/61/m²** (Vistobra Perú, tabla nov. 2026): https://vistobra.com/pe/precios/pintor

La salida `automatic_repair_scenario` contiene `illustrative_min_pen`, `illustrative_max_pen`, `items`, `excluded`, URL de fuentes, descripción y exclusiones. **`quote.total_pen` permanece `null`** porque no hay un metrado ni presupuesto confirmado. Las grietas escalonadas/posiblemente estructurales se excluyen: precisan inspección profesional. Las detecciones de pintura experimental tampoco entran en el rango.

**Ejemplo del primer informe mostrado:** una fisura de fachada con confianza 74 % y otra del 36 % produce una sola partida de sellado superficial exterior, **S/23,02–69,06 como ejemplo condicionado a que sea únicamente fisura del revestimiento**; no es el costo total de reparar esa vivienda. En informes previos, el backend calcula el escenario sin volver a subir fotos. El PDF lo incluye con sus advertencias.


## Corrección: valor de edificación en soles (14 noviembre 2026)

Tras subir 1–2 fotos, el informe ya presenta una **SIMULACIÓN DIDÁCTICA AUTOMÁTICA** en soles para explicar el cálculo; **no es el precio de la vivienda fotografiada**. El ejemplo predeterminado es Sierra, 60 m² y categoría D uniforme, igual a **S/ 61 642,20**, conforme a la tabla de valores unitarios de edificación CAP **noviembre 2026**.

Para obtener un monto referido al inmueble analizado, introduce tres datos mínimos en la misma pantalla: **área techada real en m², zona aplicable (Lima/Callao, Costa, Sierra o Selva), categoría constructiva de simulación (C–F)**. La API `POST /api/cases/{id}/official-construction-reference` calcula la suma de las siete partidas oficiales × área, guarda el resultado y lo muestra en el mismo PDF; el catálogo está en `GET /api/official-construction-values`. Puedes volver a abrir el informe desde un expediente anterior sin volver a subir fotos.

**Limitación esencial:** se usa una sola categoría para las siete partidas como escenario simplificado, **sin que YOLO verifique esa categoría**. Una tasación oficial exige categorías correctas **por partida**, área medida, depreciación, verificación de ubicación, valor del terreno y comparables. El monto visualizado **no es valor comercial, importe de garantía ni crédito sugerido** y no cambia por el mero número de fisuras detectadas. Las dos fotos no permiten calcular metros cuadrados reales por sí solas.

Fuente: cuadros oficiales publicados por el [Colegio de Arquitectos del Perú](https://cap.org.pe/valores-unitarios-oficiales-de-edificaciones/) con vigencia del 1 al 30 de noviembre de 2026. Los importes deben actualizarse cuando cambien las tablas.

# Tasación Visual Rápida — Asistente para microcrédito hipotecario o prendario

## Inspección multiclase y tasación — actualización 2026-11-13

La aplicación ahora dispone de dos detectores ONNX para CPU:

- **YOLOv8n fisuras** (OpenSistemas), verificado con inferencia real, respaldo automático.
- **YOLO26n defectos de construcción** (IAAC MaAI, licencia AGPL-3.0), exportado a ONNX y probado con inferencia CPU real: `crack`, `stairstep_crack`, `mold`, `peeling_paint`, `water_seepage`. Se habilita mediante `YOLO_DEFECT5_ENABLED=1` y requiere `YOLO_CPU_ENABLED=1`. Pesos ONNX en GitHub Release `v0.2-defect5-onnx`, con verificación de checksum antes de cargarlos. [Tarjeta y métricas originales](https://huggingface.co/iaac-maai/day4-team1-building-defect-detector).

**Precisión publicada por el autor (AP50, conjunto de prueba):** fisura 0,93; grieta escalonada 0,77; moho 0,29; filtración 0,28; pintura desprendida 0,05. Son cifras del conjunto del autor, **no** de viviendas peruanas ni de esta integración. Las categorías de superficies pueden fallar con frecuencia; los resultados son **candidatos** y requieren revisión humana.

### Qué puede y qué no puede detectar

| Elemento | Estado |
|---|---|
| Fisuras y grietas escalonadas | Modelo multiclase disponible; requiere validación local |
| Posible moho, pintura desprendida y manchas de filtración | Disponible como alerta de baja confianza validada por persona |
| Pintura general deteriorada, pisos rotos y baldosas fracturadas | Clases definidas, **sin modelo entrenado** |
| Puertas visibles o huecos sin puerta | Pendiente de dataset, entrenamiento y pruebas; jamás inferir ausencia por falta de detección |
| Paredes tarrajeadas/sin tarrajear, ladrillo visto, techos de calamina/concreto | Pendiente de entrenamiento de materiales y acabados |
| Diagnóstico de seguridad estructural y daños ocultos | No deducible de una o dos fotografías |

`scripts/train_missing_categories.py` prepara el entrenamiento cuando se disponga de fotografías etiquetadas, permiso de uso, particiones independientes y control de calidad. El script **no implica que esas clases ya se reconozcan**.

### Plan de mejoras y estimación económica

Tras subir 1–2 fotos, el informe muestra detecciones, localización por vista, confianza, prioridad de inspección y **plan de mejoras sugeridas**. La cotización automática muestra partidas/precios unitarios únicamente si existe un catálogo verificado (`TASACION_PRICE_CATALOG`) con fuente. No se inventan metros lineales/cuadrados a partir de una fotografía.

Para obtener una cantidad en soles, el panel ofrece **escenario monetario documentado** (no avalúo comercial) a partir de área construida real, costo/valor unitario de referencia y fuente documentada (con opción de incluir terreno). El cálculo y fuente se guardan en el PDF y en el JSON del expediente. En Perú pueden consultarse los [valores unitarios oficiales de edificación para 2026 del MVCS](https://www.gob.pe/institucion/vivienda/normas-legales/7568068-015-2025-vivienda-vmvu-dgprvu), pero debe seleccionarse la región y las partidas efectivamente aplicables.

**No equivale a tasación hipotecaria, costo exacto de rehabilitación ni aprobación de crédito**. Para una tasación válida faltan metraje certificado, ubicación verificada, documentación, condiciones estructurales, antigüedad, depreciación y comparables de mercado.


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
