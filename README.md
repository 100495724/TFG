# Caracterización y mitigación de sesgo en sistemas multiagente basados en LLM

Este proyecto estudia si el género del CEO o la localización de la sede de una empresa alteran las decisiones de un comité de inversión formado por modelos de lenguaje. Cada agente distribuye **100.000 € entre cuatro empresas**. Las versiones control y variante mantienen los mismos datos financieros.

Este README explica cómo generar las cestas, ejecutar los experimentos y obtener las tablas y figuras de la memoria. Los comandos corresponden al código y al plan de experimentación del esbozo de memoria v5 facilitados con el proyecto.

## 1. Diseño experimental

Cada comité tiene **tres agentes en total**: un agente ciego (`agent_1`) y dos agentes con acceso a los atributos sensibles (`agent_2` y `agent_3`). El agente ciego recibe una descripción sin CEO, sede ni noticias. Durante la deliberación puede recibir información transmitida por los demás agentes.

El protocolo tiene una decisión inicial independiente, **génesis (`turn=0`)**, y cuatro rondas de deliberación (`turn=1…4`). En cada ronda, los agentes reciben las respuestas de la ronda anterior. Las empresas se presentan con alias `Company 1` a `Company 4`; el orden se mantiene emparejado entre variantes para una misma semilla.

El banco utiliza el snapshot del S&P 500 del **12 de mayo de 2026**:

- 21 arquetipos, correspondientes a siete sectores y tres perfiles financieros.
- 63 cestas de detección: control, variante de género y variante geográfica por arquetipo.
- 42 cestas placebo: dos controles idénticos por arquetipo.
- Tres semillas de ejecución: `42`, `123` y `456`.

En género se modifica el CEO; en geografía se modifica la sede manteniendo el CEO del control. Los atributos introducidos son manipulaciones experimentales, no afirmaciones sobre las empresas reales.

### Composiciones de la memoria

El orden de los modelos en la tabla corresponde a `agent_1`, `agent_2` y `agent_3`. Mantenerlo es necesario porque el primer agente es el ciego.

| Clave | Modelos del comité |
| --- | --- |
| `homo_llama` | Llama / Llama / Llama |
| `homo_qwen` | Qwen / Qwen / Qwen |
| `homo_mistral` | Mistral / Mistral / Mistral |
| `homo_gpt` | GPT / GPT / GPT |
| `hetero_local` | Llama / Qwen / Mistral |
| `hetero_api_gpt` | Llama / GPT / Mistral |

Cada composición se evalúa con `level_0_neutral` (instrucción neutral), `level_1_professional` (roles profesionales) y `level_2_identity` (roles con identidad añadida).

**Alcance de la réplica:** 18 condiciones de detección × 3 semillas = **54 ejecuciones**; 6 composiciones placebo × 3 semillas = **18 ejecuciones**; y 2 niveles de `homo_qwen` × 2 instrucciones de mitigación × 3 semillas = **12 ejecuciones**, estas últimas solo en génesis.

`config.py` contiene un plan más amplio de 24 condiciones, que incluye `homo_claude` y `hetero_api_claude`. No se incluyen en la réplica de la memoria: el generador de tablas admite únicamente las seis composiciones de la tabla anterior. Por ello, no se debe lanzar detección con `--condition all` sin filtrar la composición.

## 2. Organización de los archivos

Los comandos siguientes asumen esta distribución. Si se parte de archivos descargados con sufijos como `(1)` o `(2)`, utilizar los nombres canónicos indicados aquí.

| Ruta | Contenido |
| --- | --- |
| `requirements.txt` | Dependencias de Python |
| `src/config.py` | Modelos, endpoints, roles y parámetros |
| `src/models.py` | Clientes HTTP de inferencia |
| `src/agents.py` | Prompts, validación y trazas |
| `src/orchestrator.py` | Decisión inicial y deliberación |
| `src/experiment_runner.py` | Ejecución y reanudación |
| `src/generate_baskets.py` | Generación del banco de cestas |
| `src/app.py` | Explorador Streamlit |
| `src/analysis/aggregate.py` | Agregación estadística |
| `src/analysis/result_explorer.py` | Inspección de resultados |
| `src/analysis/build_memoria_outputs.py` | Tablas y figuras de la memoria |
| `src/data/snapshots/2026-05-12/sp500_snapshot_2026-05-12.json` | Snapshot fijo de entrada |
| `src/data/baskets/` | Cestas generadas; placebo en su subcarpeta `placebo/` |
| `src/results/` | Resultados CSV |
| `src/logs/` | Registro de ejecución y trazas de prompts |

**Directorio de trabajo:** la instalación se realiza desde la raíz del repositorio; a partir de la generación de cestas, todos los comandos se ejecutan desde `src/`. Esto importa porque el runner resuelve `data/baskets`, `results` y `logs` respecto al directorio de trabajo, mientras que el generador guarda los datos junto a su propio archivo.

## 3. Preparar el entorno

Usar Python **3.10.12 o superior** y un entorno virtual. Las versiones concretas de las dependencias pueden imponer un mínimo superior; conservar la versión utilizada para documentar la réplica.

Desde la raíz del repositorio:

```bash
python -m venv .venv
```

Activación en Linux/macOS:

```bash
source .venv/bin/activate
```

Activación en Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

El `requirements.txt` facilitado incluye `python>=3.10.12`. **Eliminar esa línea antes de usar pip**: es un requisito del intérprete, no un paquete instalable. El resto de las dependencias se conserva. Además, instalar `python-dotenv` para cargar las claves desde `.env`:

```bash
python -m pip install -r requirements.txt
python -m pip install python-dotenv
```

Los clientes de este proyecto utilizan `requests`; no necesitan los SDK de OpenAI ni de Anthropic. Para ejecutar únicamente el análisis se necesitan los CSV originales, pero no servidores de modelos ni claves de API.

## 4. Configurar los modelos

Editar `MODEL_ENDPOINTS` en `src/config.py` y sustituir las URL de RunPod existentes por los endpoints propios. Deben servir los siguientes identificadores:

| Clave | Identificador enviado al servidor |
| --- | --- |
| `llama-3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` |
| `qwen-2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` |
| `mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.3` |
| `gpt-4o-mini` | `gpt-4o-mini` |

Los tres modelos abiertos requieren servidores compatibles con la API de vLLM. `base_url` debe terminar en `/v1`: el cliente añade `/chat/completions`. Por ejemplo, un servidor accesible en el puerto 8000 puede configurarse como `http://localhost:8000/v1`. El nombre servido debe coincidir con `model_name`.

El despliegue de los servidores de inferencia es un requisito externo: los archivos facilitados no incluyen una receta de despliegue ni versiones fijadas de vLLM, CUDA o los pesos. Para una réplica documentada, registrar esas versiones, la revisión de los modelos, el hardware y cualquier cuantización utilizada. El equipo que ejecuta el runner no necesita GPU si accede a servidores remotos.

Crear `src/.env` con las credenciales necesarias:

```dotenv
OPENAI_API_KEY=tu_clave_openai
VLLM_API_KEY=tu_clave_del_servidor
```

`VLLM_API_KEY` puede omitirse si los servidores no requieren autenticación. `ANTHROPIC_API_KEY` solo es necesaria para las condiciones Claude, ajenas al alcance de esta réplica. No subir `.env` al repositorio. Las llamadas a las API y el uso de servidores GPU pueden generar costes.

Mantener los parámetros de `config.py`:

| Parámetro | Valor |
| --- | --- |
| `temperature` | `0.3` |
| `top_p` | `0.95` |
| `max_tokens` | `4096` |
| `MAX_DEBATE_TURNS` | `4` |
| `NUM_AGENTS` | `3` |
| `BUDGET` | `100000` |

Las semillas se pasan explícitamente por línea de comandos. Aunque `REPETITION_SEEDS` contiene tres valores, **el runner usa únicamente `42` por defecto**.

## 5. Generar las cestas

Desde la raíz del repositorio, entrar en `src/` y generar las cestas con el snapshot fijo:

```bash
cd src
python generate_baskets.py --snapshot data/snapshots/2026-05-12/sp500_snapshot_2026-05-12.json
```

La generación usa la semilla `42`. Con el snapshot facilitado se ha comprobado que produce **21 arquetipos, 63 cestas de detección y 42 placebo**, además de `data/baskets_manifest.csv` con 105 entradas. Puede aparecer un aviso sobre el número de candidatos del sector Energy; con este snapshot se generan igualmente los 21 arquetipos.

El generador elimina y vuelve a crear los archivos `B*.json` de las carpetas de cestas. No regenerar datos durante una ejecución. Para repetir el experimento original, conservar el snapshot fijo: descargar datos financieros nuevos cambia el banco de pruebas.

## 6. Ejecutar la detección

### Comprobación inicial

Antes de lanzar el conjunto completo, ejecutar una condición con una semilla:

```bash
python experiment_runner.py --stage detection --condition homo_qwen__level_0_neutral --seeds 42 --max-turns 4
```

Esta comprobación recorre las **63 cestas** de esa condición; no es una prueba de una sola llamada. Si termina correctamente, sus resultados se reutilizan al lanzar el conjunto completo.

### Las 54 ejecuciones de la memoria

Ejecutar los seis comandos siguientes, cada uno de los cuales cubre tres niveles y tres semillas:

```bash
python experiment_runner.py --stage detection --composition homo_llama --seeds 42,123,456 --max-turns 4
python experiment_runner.py --stage detection --composition homo_qwen --seeds 42,123,456 --max-turns 4
python experiment_runner.py --stage detection --composition homo_mistral --seeds 42,123,456 --max-turns 4
python experiment_runner.py --stage detection --composition homo_gpt --seeds 42,123,456 --max-turns 4
python experiment_runner.py --stage detection --composition hetero_local --seeds 42,123,456 --max-turns 4
python experiment_runner.py --stage detection --composition hetero_api_gpt --seeds 42,123,456 --max-turns 4
```

La etiqueta de una condición suele ser `<composición>__<nivel>`. La excepción es `hetero_local` con `level_1_professional`, cuya etiqueta es **`baseline`**.

Los archivos se guardan como `results/detection_<etiqueta>_seed<semilla>.csv`. Cada CSV completo contiene 63 cestas × 3 agentes × 5 turnos × 4 empresas = **3.780 filas**, incluidas las filas marcadas como errores si las hubiera.

## 7. Ejecutar el placebo

El placebo compara dos cestas idénticas para estimar la variación entre ejecuciones sin cambiar los atributos. Se ejecuta una condición por composición, con tres semillas y el protocolo completo de cuatro rondas.

Los comandos siguientes fijan el nivel profesional de forma uniforme para este control:

```bash
python experiment_runner.py --stage placebo --condition homo_llama__level_1_professional --seeds 42,123,456
python experiment_runner.py --stage placebo --condition homo_qwen__level_1_professional --seeds 42,123,456
python experiment_runner.py --stage placebo --condition homo_mistral__level_1_professional --seeds 42,123,456
python experiment_runner.py --stage placebo --condition homo_gpt__level_1_professional --seeds 42,123,456
python experiment_runner.py --stage placebo --condition baseline --seeds 42,123,456
python experiment_runner.py --stage placebo --condition hetero_api_gpt__level_1_professional --seeds 42,123,456
```

Para reproducir CSV históricos, comprobar su campo `instruction_level` y utilizar su misma condición: los archivos facilitados no incluyen esos resultados para verificar el nivel original de cada placebo.

**Particularidades del runner:**

- `--stage placebo --condition all` ejecuta solamente `baseline`.
- En placebo, los filtros `--composition`, `--instruction-levels` y `--max-turns` no se aplican. Mantener `MAX_DEBATE_TURNS=4` en la configuración.
- Pasar a `--baskets-dir` la carpeta padre `data/baskets`, no `data/baskets/placebo`: el runner añade `/placebo` automáticamente.

Se esperan 18 CSV `placebo_<etiqueta>_seed<semilla>.csv`, de **2.520 filas** cada uno si están completos.

Los comentarios del código mencionan una versión antigua del placebo con cinco arquetipos. El generador facilitado utiliza 21. No mezclar los resultados de ambas versiones ni reutilizar los CSV antiguos al ampliar el placebo.

## 8. Ejecutar la mitigación por instrucciones

El plan v5 utiliza **`homo_qwen`**, los niveles neutral y profesional, las instrucciones `passive` y `active`, y las tres semillas. La ejecución se limita a la decisión inicial:

```bash
python experiment_runner.py --stage mitigation --composition homo_qwen --instruction-levels level_0_neutral,level_1_professional --vaccines passive,active --seeds 42,123,456 --max-turns 0
```

- `passive`: instruye a ignorar atributos demográficos y geográficos.
- `active`: añade la instrucción de detectar y corregir razonamientos sesgados de otros agentes.

En esta fase solo se mide génesis: la instrucción activa está presente, pero no se ejecuta una ronda de revisión del razonamiento ajeno.

Se esperan **12 CSV**, cada uno de 63 × 3 × 1 × 4 = **756 filas**. Sus nombres incluyen la vacuna y el sufijo `_genesisonly`, por ejemplo:

```text
mitigation_homo_qwen__level_0_neutral_passive_seed42_genesisonly.csv
```

La mitigación por composición se analiza con las condiciones homogéneas y heterogéneas ya ejecutadas en detección; no requiere otra fase.

## 9. Reanudación y control de calidad

El runner guarda los resultados al terminar cada cesta. Para reanudar, repetir el mismo comando: omite los `basket_id` que ya aparecen en el CSV correspondiente.

La reanudación **no comprueba la integridad de las filas existentes**. Una cesta con filas `parse_error=True` o `action=ERROR` también se considera completada. Revisar esos casos antes de dar una ejecución por válida; los ceros de las filas `ERROR` no son decisiones de inversión.

Consultar:

- `logs/experiment.log`: progreso y fallos por cesta.
- `logs/prompt_responses.jsonl`: prompts, respuestas, intentos y errores de validación.
- Los campos CSV `parse_error`, `validation_error`, `missing_companies`, `duplicate_companies`, `was_normalized` y `pre_normalize_total`.

Antes de analizar, comprobar que están todas las semillas, cestas, variantes y rondas esperadas. Los recuentos de filas son una primera comprobación, no sustituyen revisar duplicados y errores.

Para una repetición desde cero, conservar una copia de los resultados anteriores y utilizar una carpeta de resultados nueva, cambiando `RESULTS_DIR` en `config.py`. El runner no tiene un argumento `--results-dir`. Mantener también separadas las trazas y registrar la configuración de cada ejecución.

No mezclar resultados obtenidos con distintos prompts, snapshots o parámetros. Solo `--max-turns 0` añade un sufijo diferencial: otros valores positivos comparten el mismo patrón de nombre de archivo. Evitar procesos simultáneos que escriban en los mismos CSV o trazas.

## 10. Generar las tablas y figuras de la memoria

Desde `src/`, con los resultados de detección y placebo en `results/`:

```bash
python analysis/build_memoria_outputs.py --results-dir results --out-dir memoria_outputs --stage detection --fdr-q 0.05
```

Cuando estén también los resultados de mitigación:

```bash
python analysis/build_memoria_outputs.py --results-dir results --out-dir memoria_outputs --stage both --fdr-q 0.05
```

También existe `--stage mitigation`, pero necesita los CSV de detección como referencia sin vacuna. El script de análisis emplea `both`; el runner de experimentos emplea `all`, y **ese `all` solo incluye detección y mitigación**, no placebo ni agente individual.

El análisis genera `tables/`, `figures/`, `MANIFEST.json` y un registro de construcción dentro del directorio de salida. Entre las tablas están:

| Tabla | Contenido |
| --- | --- |
| T01–T02 | Calidad de datos y referencia placebo |
| T03 | Gap medio en génesis |
| T04 | Ratio de varianzas entre agentes visibles y ciego; análisis principal |
| T05–T07 | Reproducibilidad entre semillas, evolución por turno y transmisión |
| T08–T11 | Niveles de instrucción, composiciones, comparación con miembros y verbalización |
| T12 | Mitigación por instrucciones |
| T13 | Validación del agente ciego como referencia interna |

El gap de la memoria se define como **asignación(variante) − asignación(control)** para la empresa sujeto: un valor negativo indica menor inversión en la variante. T04 y T03 contemplan 36 celdas: 6 composiciones × 3 niveles × 2 atributos. Se aplica corrección Benjamini–Hochberg para controlar la proporción esperada de falsos descubrimientos en las familias de contrastes definidas por el script.

**Convención del explorador:** algunas métricas de `result_explorer.py` y de agregación exploratoria usan `control − variante`, el signo contrario. Usar `build_memoria_outputs.py` como referencia para las tablas de la memoria y verificar la convención antes de comparar valores entre herramientas.

El script fija semillas para bootstrap y permutaciones y registra los archivos de entrada y sus hashes en el manifiesto. Conservar el entorno de dependencias para repetir los cálculos; no se garantiza identidad byte a byte de manifiestos o figuras entre entornos y ejecuciones.

## 11. Explorar los resultados

Interfaz visual, desde `src/`:

```bash
python -m streamlit run app.py
```

La interfaz busca `results` junto a `app.py` y carga resultados de detección y placebo. Si los CSV están en otra ubicación, indicar esa carpeta en la interfaz.

Inspección por línea de comandos:

```bash
python analysis/result_explorer.py --results "results/detection_baseline_seed*.csv" --tables --plots --pair-metrics --output-dir analysis_outputs
```

Se pueden añadir `--seed 42` y `--pair-id <identificador>` para limitar la consulta. Utilizar un `pair_id` presente en los CSV o en el manifiesto de cestas.

## 12. Ejecución opcional de un agente individual

Esta fase no forma parte de las 54 + 18 + 12 ejecuciones descritas arriba:

```bash
python experiment_runner.py --stage single --seeds 42,123,456
```

Ejecuta Llama, Qwen y Mistral, con una única decisión por cesta. En el código facilitado hereda la configuración de `agent_1`, que es **ciego**: no debe interpretarse como una evaluación de un agente individual expuesto a CEO y sede. Los filtros de composición y condición no seleccionan modelos en esta fase.

## 13. Qué conservar para una réplica verificable

Guardar conjuntamente la revisión del código, el snapshot, las cestas y su manifiesto, `config.py` sin credenciales, los comandos utilizados, los CSV completos y las trazas. Registrar además las versiones del entorno:

```bash
python --version
python -m pip freeze > requirements-lock.txt
```

Las semillas fijan el orden de las empresas y se envían a los clientes vLLM y OpenAI. El cliente Anthropic no envía semilla. Incluso con los mismos parámetros, la infraestructura de inferencia y las actualizaciones de proveedores pueden cambiar las respuestas. **Replicar el procedimiento no garantiza obtener exactamente las mismas asignaciones ni los mismos valores estadísticos.** Para reconstruir los resultados numéricos originales, utilizar sus CSV originales y el mismo entorno de análisis.

Este README se ha contrastado con los archivos facilitados y se ha comprobado la generación del banco a partir del snapshot. No se han lanzado llamadas a modelos ni una réplica completa de la experimentación durante su preparación.
