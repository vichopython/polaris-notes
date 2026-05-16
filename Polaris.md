# Polaris

### Plan de carrera en IA aplicada · v1.0 · Reemplaza a Protocolo Helio

> *Pensado para durar 18–24 meses de estudio activo, y dejarte herramientas que sigan vigentes en 10–20 años.*

---

## Índice

0. [Filosofía: lo que dura vs lo que caduca](#0-filosofía-lo-que-dura-vs-lo-que-caduca)
1. [Tu ventaja injusta](#1-tu-ventaja-injusta)
2. [Estructura general del plan](#2-estructura-general-del-plan)
3. [Pilar 0 — Setup técnico y mental](#pilar-0--setup-técnico-y-mental)
4. [Pilar 1 — Pensamiento computacional](#pilar-1--pensamiento-computacional)
5. [Pilar 2 — Python real](#pilar-2--python-real)
6. [Pilar 3 — Matemática para IA](#pilar-3--matemática-para-ia)
7. [Pilar 4 — Datos: SQL, Pandas, estadística](#pilar-4--datos-sql-pandas-estadística)
8. [Pilar 5 — Machine Learning clásico](#pilar-5--machine-learning-clásico)
9. [Pilar 6 — Deep Learning y LLMs por dentro](#pilar-6--deep-learning-y-llms-por-dentro)
10. [Pilar 7 — Aplicaciones de IA en producción](#pilar-7--aplicaciones-de-ia-en-producción)
11. [Pilar 8 — Software engineering durable](#pilar-8--software-engineering-durable)
12. [Pilar 9 — Vertical, portfolio y posicionamiento](#pilar-9--vertical-portfolio-y-posicionamiento)
13. [Hábitos transversales](#hábitos-transversales)
14. [Herramientas: qué pagar, qué no](#herramientas-qué-pagar-qué-no)
15. [Hitos cada 6 meses](#hitos-cada-6-meses)
16. [Cómo medir progreso real](#cómo-medir-progreso-real)
17. [Errores que cuestan meses](#errores-que-cuestan-meses)
18. [Biblioteca permanente](#biblioteca-permanente)
19. [Renovación del plan](#renovación-del-plan)

---

## 0. Filosofía: lo que dura vs lo que caduca

Antes de la ruta, los principios. Si entiendes esto, puedes rearmar el plan tú mismo cuando algo del mercado cambie.

### Lo que dura décadas (alta prioridad)

- **Matemática**: álgebra lineal, probabilidad, cálculo básico. Igual hoy que hace 100 años.
- **Estadística inferencial**: la base de todo modelo predictivo desde Fisher.
- **Algoritmos y estructuras de datos**: las ideas no cambian. Una lista enlazada en 1970 es la misma de hoy.
- **SQL**: 50 años de existencia, sigue siendo el lenguaje estándar para datos estructurados.
- **Linux/Unix**: 50+ años, corre el 96% de los servidores del mundo.
- **Git**: el día que lo reemplacen, la idea de control de versiones seguirá. Aprender Git te da la idea.
- **Python**: existe desde 1991, sigue creciendo, no se va a ningún lado.
- **Comunicación escrita clara**: ningún LLM te quita el valor de pensar y escribir bien.
- **Conocimiento de dominio** (en tu caso: negocio, operación, finanzas).

### Lo que dura 5–15 años (prioridad media)

- **PyTorch / TensorFlow**: cambiarán de versión pero el paradigma probablemente no.
- **Conceptos de LLMs**: embeddings, attention, fine-tuning, RAG. Las implementaciones evolucionarán; la teoría base es estable.
- **Docker y contenedores**: el concepto es eterno; las herramientas pueden rotar.
- **Arquitecturas en la nube** (la idea, no el proveedor específico).

### Lo que caduca en 1–3 años (úsalo, no te ancles)

- **Frameworks específicos de aplicaciones LLM** (LangChain, LlamaIndex en sus formas actuales). Hoy útiles, mañana pueden ser el jQuery del 2010.
- **Modelos específicos**: Claude 4.7 hoy, otro mañana.
- **Herramientas de productividad** (Cursor, Copilot): cambiarán de marca pero la idea de IDE con IA es permanente.
- **Términos de moda**: "prompt engineering" como disciplina separada probablemente desaparezca; la habilidad de comunicarse claramente con un modelo, no.

### Regla operativa del plan

> **Cuando algo es volátil, aprende el concepto debajo, no la sintaxis.**
> **Cuando algo es durable, sí dedícale tiempo a memorizarlo.**

Esa regla aplica a cada módulo de aquí en adelante.

---

## 1. Tu ventaja injusta

Eres ingeniero comercial con 18–24 meses de aprendizaje técnico por delante. Eso no te convierte en ML engineer puro — y está bien, no querés competir ahí. Te convierte en:

**Implementador de IA en negocios**: el perfil que entiende qué problema vale la pena resolver (lado comercial), traduce eso a una solución técnica (lo que vas a aprender), y la lleva a producción.

En el mercado actual de Chile, Latinoamérica y Australia: este perfil está mejor pagado y es más escaso que un ML engineer junior. Las empresas no saben qué problema resolver con IA — no les falta gente que sepa entrenar modelos genéricos. Les falta alguien que entienda cuándo no usar IA, cuándo sí, qué problema vale el costo, y cómo medir si funcionó.

Tu CV objetivo a 24 meses:

> *Ingeniero Comercial · Implementación de IA en procesos de negocio · Especialización en automatización de operaciones (HR/finanzas) con LLMs y machine learning aplicado.*

No es "AI Engineer". No es "Data Scientist". Es algo más raro y más útil.

---

## 2. Estructura general del plan

10 pilares (0 al 9). Cada uno tiene:
- **Objetivo**: qué sabrás hacer al terminar.
- **Por qué dura**: justificación de longevidad.
- **Recursos**: gratis primero, pagados solo cuando se justifica.
- **Proyecto entregable**: lo que construyes para demostrarte que lo aprendiste.
- **Criterio de salida**: cómo sabes que terminaste.
- **Tiempo estimado** (a ~8–10 horas semanales).

**Secuencia recomendada**: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9.

**Paralelizables**: una vez en el Pilar 4, puedes correr el Pilar 9 (portfolio) en paralelo permanente.

**Tiempo total estimado**: 18–22 meses si haces 8–10 horas semanales con consistencia. Si haces 4–6, son 28–32 meses. Tu ritmo es tu ritmo.

---

## Pilar 0 — Setup técnico y mental

**Duración**: 1–2 semanas.

### Objetivo
Tener un entorno de trabajo limpio, una rutina sostenible y los fundamentos de uso de la máquina antes de aprender nada serio.

### Por qué dura
Las herramientas pueden cambiar pero la disciplina de tener tu entorno bajo control no. Esto se hace una vez bien y dura años.

### Recursos
1. **Configurar VS Code** con extensiones esenciales:
   - Python (Microsoft)
   - GitLens
   - Markdown All in One
   - una de Claude o Continue.dev para asistencia (Cursor más adelante).
2. **Crear cuenta en GitHub** si no la tienes activa. Username profesional (tu nombre real o algo serio).
3. **Terminal**: aprende lo básico de la línea de comandos. Recurso gratis: *The Missing Semester of Your CS Education* del MIT — clases 1 ("Shell") y 2 ("Tools and Scripting"). Está en YouTube y en su web (missing.csail.mit.edu).
4. **Git básico**: "Git & GitHub Crash Course" de freeCodeCamp en YouTube (1 hora aprox). Practica clonar un repo, hacer commit, push, pull, branch.
5. **Hábito de estudio**: define cuándo y dónde estudias. La regularidad importa más que la duración. 5 días × 1.5 horas vence a 1 día × 8 horas.

### Proyecto entregable
Un repositorio en GitHub llamado `polaris-notes` donde vas a llevar apuntes en markdown de todo lo que aprendas en los próximos 24 meses. Tu primer commit: este mismo archivo (Polaris.md).

### Criterio de salida
- Puedes abrir terminal, navegar carpetas, crear archivos, ejecutar comandos básicos sin mirar Google cada paso.
- Puedes clonar un repo, hacer un cambio, commit, y push sin pánico.
- Tu VS Code está configurado y cómodo.

---

## Pilar 1 — Pensamiento computacional

**Duración**: 2–3 meses.

### Objetivo
Aprender a descomponer problemas, pensar algorítmicamente, y entender cómo funciona una computadora por dentro. No es Python específico — es el músculo mental sobre el que toda la programación se apoya.

### Por qué dura
Pensamiento computacional es una habilidad meta. Los lenguajes cambian, los frameworks mueren, los modelos pasan; la capacidad de descomponer un problema en subproblemas tratables es eterna. CS50 lleva 15+ años siendo el mejor curso introductorio del mundo. No va a envejecer mal.

### Recursos
1. **CS50 de Harvard** (gratis, en edX o YouTube oficial). Es duro. Vale cada hora. Cubre C, Python, SQL, web, todo conceptualmente. Aproximadamente 10–12 semanas.
   - Si la dificultad de la primera semana (en C) te frena, salta directo a **CS50P (CS50's Introduction to Programming with Python)**, también gratis. Más amable, te lleva al mismo lugar.
2. Complemento opcional: *Grokking Algorithms* de Aditya Bhargava. Libro corto, visual, gentil. Algoritmos básicos sin sufrir.

### Proyecto entregable
El proyecto final de CS50 o CS50P. Lo subes a tu GitHub con README explicando qué hace y cómo correrlo.

### Criterio de salida
- Lees código Python básico y entiendes qué hace.
- Puedes escribir una función que resuelve un problema simple (ordenar una lista, contar palabras en un texto, etc.) sin mirar ejemplos.
- Sabes qué es una variable, un loop, una condición, una función, una lista, un diccionario, una clase (al menos qué significa).
- Has visto qué es la complejidad algorítmica (Big O) aunque no la domines.

---

## Pilar 2 — Python real

**Duración**: 3–4 meses.

### Objetivo
Pasar de "sé lo básico de Python" a "puedo construir cosas reales en Python sin sufrir". Esto es el nivel mínimo profesional.

### Por qué dura
Python tiene 35 años, sigue creciendo, y es el idioma franco de IA, datos y automatización. La sintaxis no va a cambiar. Lo que aprendas hoy te sirve en 2040.

### Recursos
1. **Angela Yu — 100 Days of Code** (ya lo tienes comprado). Haz hasta el día 60 mínimo. Los últimos 40 son web/data más selectivos, no obligatorios.
2. **Complementar con la documentación oficial de Python** cuando algo no quede claro. docs.python.org/3/tutorial. Acostúmbrate a leer documentación oficial — es una habilidad en sí.
3. **Real Python** (realpython.com). Algunos artículos son gratis, otros pagados. Los gratis son excelentes para profundizar temas puntuales.
4. **Libro de referencia**: *Python Crash Course* de Eric Matthes, 3ra edición. Como libro de cabecera para consultar. No hace falta leerlo entero.

### Proyecto entregable
Construye **dos proyectos personales** en Python, no tutoriales:
1. **Una automatización útil de tu trabajo en HR**: por ejemplo, una versión mejorada del lector de turnos que hiciste para BUK, con tests, documentación y un README profesional.
2. **Algo que te interese personalmente**: un script que tracquea tu progreso de gym, un bot de Telegram que te manda cosmología cada mañana, un analizador de tus gastos. Cualquier cosa que te motive.

Ambos en GitHub, con README, con instrucciones para correrlos.

### Criterio de salida
- Puedes escribir un script de 200 líneas que hace algo útil sin mirar tutoriales paso a paso.
- Entiendes funciones, clases, manejo de archivos, manejo de errores (try/except), módulos, entornos virtuales (`venv`), instalación de paquetes (`pip`).
- Puedes leer un README de un proyecto open source y entender qué hace.
- Has usado al menos: `requests`, `pandas`, `openpyxl` o similares.

---

## Pilar 3 — Matemática para IA

**Duración**: 2–3 meses.

### Objetivo
Tener la intuición matemática suficiente para entender qué pasa dentro de un modelo de ML. **No es para volverte matemático**. Es para que cuando leas "loss function", "gradiente", "vector de embeddings" o "matriz de pesos" no te suene a chino.

### Por qué dura
La matemática de los modelos de hoy es la misma de hace 80 años. El backpropagation se inventó en los 70s. El álgebra lineal en el 1800s. La probabilidad en el 1600s. Esto literalmente no envejece.

### Recursos
1. **3Blue1Brown — Essence of Linear Algebra** (YouTube, gratis). Serie de 15 videos. Mira los 15. Es la mejor introducción visual al álgebra lineal que existe.
2. **3Blue1Brown — Essence of Calculus** (YouTube, gratis). Misma serie, otro tema. Solo lo necesario: derivadas, gradiente.
3. **StatQuest with Josh Starmer** (YouTube, gratis). Sección "Statistics Fundamentals". Para probabilidad y estadística.
4. **Mathematics for Machine Learning** de Imperial College en Coursera. Auditable gratis. Tres cursos:
   - Linear Algebra
   - Multivariate Calculus
   - PCA
5. **Khan Academy** (gratis) para reforzar cualquier cosa que se sienta floja.

### Proyecto entregable
Un notebook de Jupyter en tu GitHub donde implementas, **desde cero, sin librerías**:
- Multiplicación de matrices.
- Una regresión lineal simple usando solo numpy.
- Un cálculo de gradiente manual.

Documentado con explicaciones tuyas. Esto es para ti, para confirmar que entendiste, no para impresionar.

### Criterio de salida
- Entiendes qué es un vector, una matriz, un producto punto.
- Entiendes intuitivamente qué es un gradiente y por qué un modelo "baja" por él.
- Entiendes media, varianza, desviación estándar, distribución normal, probabilidad condicional, teorema de Bayes a nivel intuitivo.
- Cuando alguien dice "este modelo tiene 7 mil millones de parámetros", entiendes qué significa eso geométricamente.

---

## Pilar 4 — Datos: SQL, Pandas, estadística

**Duración**: 2–3 meses.

### Objetivo
Saber traer, limpiar, transformar y entender datos. Sin esto, todo lo de IA es vapor.

### Por qué dura
SQL fue diseñado en 1974 y todavía es el estándar. Pandas existe desde 2008 y no muestra signos de morir. La estadística aplicada es de los 1700s. Los formatos cambian; las ideas no.

### Recursos
1. **SQL**:
   - **SQLBolt** (gratis, sqlbolt.com): ejercicios interactivos, 1–2 semanas.
   - **Mode SQL Tutorial** (gratis, mode.com/sql-tutorial): siguiente paso después de SQLBolt.
   - **PostgreSQL Tutorial** (postgresqltutorial.com, gratis) para profundizar.
2. **Pandas**:
   - **Kaggle Learn — Pandas** (gratis, ~10h).
   - **Kaggle Learn — Data Visualization** (gratis, ~10h).
3. **Estadística aplicada**:
   - **StatQuest** completo: secciones "Statistics Fundamentals", "Probability", "Linear Regression".
   - Libro opcional: *Practical Statistics for Data Scientists* de Bruce, Bruce y Gedeck.

### Proyecto entregable
Analiza un dataset real de algo que te interese. Sugerencia personalizada: **datos públicos del INE de Chile** o **datos de la Dirección del Trabajo**. Limpia, transforma, visualiza, escribe conclusiones. Sube todo a GitHub.

Otra idea: descarga tus propios datos (Strava, Spotify wrapped, finanzas personales) y haz un análisis estadístico honesto.

### Criterio de salida
- Puedes escribir queries SQL con JOINs, GROUP BY, subqueries, window functions básicas.
- Puedes tomar un CSV sucio y dejarlo limpio en pandas en menos de una hora.
- Puedes hacer un gráfico que no sea feo.
- Sabes la diferencia entre correlación y causalidad y por qué importa.
- Puedes calcular y explicar: media, mediana, percentiles, intervalo de confianza, p-valor (con humildad).

---

## Pilar 5 — Machine Learning clásico

**Duración**: 3 meses.

### Objetivo
Entender los modelos clásicos de ML, cuándo usarlos, cómo evaluarlos, y los conceptos transversales que se aplican también al deep learning: overfitting, validación cruzada, métricas, sesgo, varianza.

### Por qué dura
Los algoritmos de este pilar (regresión lineal, logística, árboles, random forests, gradient boosting, k-means) tienen entre 60 y 200 años. Siguen siendo competitivos en muchos problemas reales. En proyectos comerciales, un XGBoost bien hecho vence a un modelo profundo mal hecho 9 de cada 10 veces.

### Recursos
1. **Andrew Ng — Machine Learning Specialization** en Coursera (tres cursos, auditable gratis). Es el estándar global. Andrew enseña como nadie. Ten paciencia con la matemática, ya la armaste en el Pilar 3.
2. **Kaggle Learn**: "Intro to Machine Learning" + "Intermediate Machine Learning". Gratis, hands-on.
3. **Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow** de Aurélien Géron (3ra ed.). Libro pago pero la primera mitad es oro. Si vas a comprar un solo libro técnico en estos dos años, que sea este.

### Proyecto entregable
Participa en una competencia "Getting Started" de Kaggle (Titanic, House Prices) y publica tu notebook. No para ganar, para terminar el ciclo completo: cargar datos → explorar → limpiar → modelar → evaluar → escribir conclusiones.

Mejor todavía: define un problema **de tu dominio comercial** y resuélvelo con ML clásico. Ejemplo: predecir rotación de empleados de un dataset público.

### Criterio de salida
- Conoces los algoritmos: regresión lineal, regresión logística, árboles, random forest, gradient boosting, k-means, PCA.
- Sabes la diferencia entre supervisado y no supervisado, clasificación y regresión.
- Entiendes overfitting y cómo combatirlo (regularización, validación cruzada, más datos).
- Sabes qué métricas usar para qué problema (accuracy vs precision vs recall vs F1 vs AUC vs MAE vs RMSE) y por qué importa.
- Puedes entrenar un modelo en scikit-learn de principio a fin sin tutorial.

---

## Pilar 6 — Deep Learning y LLMs por dentro

**Duración**: 3 meses.

### Objetivo
Entender cómo funcionan las redes neuronales profundas y, en particular, los transformers (la arquitectura detrás de todos los LLMs actuales). No para entrenar GPT-5 desde cero; para no ser una caja negra cuando trabajes con uno.

### Por qué dura
La arquitectura transformer es de 2017 y sigue siendo la base de todo. Pueden aparecer variantes (state-space models, mamba, etc.) pero el concepto de attention, embeddings y entrenamiento por backprop va a estar vigente toda tu carrera.

### Recursos
1. **fast.ai — Practical Deep Learning for Coders** (gratis, course.fast.ai). Enfoque top-down: construyes modelos antes de entender toda la matemática. Calza con tu estilo de aprendizaje.
2. **Andrej Karpathy — "Neural Networks: Zero to Hero"** (YouTube, gratis). Serie de videos donde construye redes neuronales desde cero, incluyendo un GPT mini. Si haces solo una cosa en este pilar, que sea esto. Karpathy es uno de los mejores profesores de IA del mundo.
3. **Hugging Face NLP Course** (gratis, huggingface.co/learn/nlp-course). Capítulos 1–6.
4. **Paper**: *Attention Is All You Need* (2017). Léelo dos veces. La primera no entenderás todo; la segunda, después de Karpathy, sí.
5. **Libro opcional**: *Deep Learning* de Goodfellow, Bengio, Courville. Disponible gratis online (deeplearningbook.org). Como referencia, no como lectura lineal.

### Proyecto entregable
Construye, siguiendo a Karpathy, un **nano-GPT** desde cero entrenado en algún texto que te guste (Borges, Bolaño, *Stormlight Archive* en español, lo que quieras). No será bueno. Da igual. El valor está en haber pasado por cada paso.

### Criterio de salida
- Entiendes qué es una red neuronal: capas, pesos, activación, función de pérdida, retropropagación.
- Entiendes qué es un embedding: representación vectorial de algo.
- Entiendes el mecanismo de attention al menos a nivel conceptual.
- Sabes la diferencia entre pre-training, fine-tuning y RLHF.
- Puedes leer un paper de IA y entender el abstract y la introducción.

---

## Pilar 7 — Aplicaciones de IA en producción

**Duración**: 2–3 meses.

### Objetivo
Construir aplicaciones reales con LLMs: RAG, agentes, evaluación, observabilidad. La frontera comercial actual.

### Por qué dura (con asterisco)
Este es el pilar más volátil del plan en cuanto a **herramientas**. Pero los **patrones** son durables:
- Recuperación + generación seguirá siendo el patrón base por años.
- Los agentes (LLMs que usan herramientas) son la dirección estratégica clara.
- La evaluación de sistemas de IA va a ser un campo permanente.

Por eso este pilar enfatiza **patrones conceptuales** sobre frameworks. Los frameworks los aprendes en una semana cuando los necesites.

### Recursos
1. **Anthropic Academy** (gratis, anthropic.skilljar.com). Termina Claude 101 (en curso), sigue con los cursos de API, prompt engineering, tool use y building with Claude.
2. **DeepLearning.AI Short Courses** (gratis, deeplearning.ai/short-courses). Selección crítica:
   - "Building Systems with the ChatGPT API"
   - "LangChain for LLM Application Development"
   - "LangChain: Chat with Your Data"
   - "Building and Evaluating Advanced RAG"
   - "Functions, Tools and Agents with LangChain"
   - "AI Agents in LangGraph"
   - "Evaluating and Debugging Generative AI"
3. **Eugene Yan blog** (eugeneyan.com) y **Chip Huyen blog** (huyenchip.com): dos de las mejores fuentes públicas sobre ML/LLMs en producción. Lectura permanente.

### Proyecto entregable (este es importante)
**Atlas Legal**: un RAG sobre la legislación laboral chilena. Carga el Código del Trabajo + dictámenes de la Dirección del Trabajo, indexa, expone una interfaz simple donde un asistente HR puede preguntar cosas y obtener respuestas con citas. Tiene que tener:
- Sistema de evaluación (no "parece que anda", sino métricas).
- Manejo de errores.
- README profesional.
- Análisis honesto de limitaciones.

**Este es tu proyecto bandera**. El que va a abrir puertas. Útil para ti hoy en HR, demostrable mañana en cualquier entrevista. Tómate el tiempo que requiera.

### Criterio de salida
- Puedes construir un RAG funcional con citas.
- Entiendes cómo evaluar un sistema de IA más allá de "parece que funciona".
- Has construido al menos un agente con tool use.
- Sabes qué es observabilidad en sistemas de IA (logs, traces, métricas).
- Entiendes el costo real (latencia, tokens, $) de las decisiones de diseño.

---

## Pilar 8 — Software engineering durable

**Duración**: 2–3 meses.

### Objetivo
Cerrar la brecha entre "scripts que funcionan en mi laptop" y "código que vive en producción y otros pueden mantener". Esto es lo que separa al hobbyista del profesional.

### Por qué dura
Las prácticas de este pilar tienen 30–50 años. Tests, control de versiones, modularidad, deployment: la forma cambia, los principios no.

### Recursos
1. **The Missing Semester of Your CS Education** (MIT, gratis). Las clases que no hiciste en el Pilar 0: debugging, profiling, metaprogramming, seguridad.
2. **Testing en Python**: documentación oficial de `pytest` (pytest.org/en/stable). Más un par de tutoriales de Real Python.
3. **FastAPI**: tutorial oficial (fastapi.tiangolo.com). Gratis, excelente.
4. **Docker**: "Docker for Beginners" de freeCodeCamp (YouTube, gratis). Lo esencial: imágenes, contenedores, Dockerfile, docker-compose.
5. **Cloud básico**: cuenta gratis en Railway, Render o Fly.io. Deploy real, sin tarjeta. Más tarde, AWS o GCP si tu trayectoria lo pide (no es urgente).
6. **Libro opcional pero recomendado**: *The Pragmatic Programmer* (Hunt & Thomas). No es de un lenguaje específico, es de cómo pensar como ingeniero.

### Proyecto entregable
Convierte Atlas Legal (Pilar 7) en un servicio **real**:
- API con FastAPI.
- Tests con pytest (al menos un 50% de cobertura).
- Dockerfile que lo empaqueta.
- Deployment en Railway o Render con URL pública.
- CI básico con GitHub Actions (corre los tests en cada push).

### Criterio de salida
- Sabes escribir tests y entiendes por qué son innegociables.
- Puedes empaquetar un proyecto en Docker y correrlo en cualquier máquina.
- Tienes algo desplegado con URL pública que cualquier persona puede ver.
- Entiendes qué es CI/CD aunque sea a nivel básico.
- Cuando alguien te dice "stage", "prod", "rollback", "feature flag" no te suena a brujería.

---

## Pilar 9 — Vertical, portfolio y posicionamiento

**Duración**: continuo desde el Pilar 4 en adelante. Se intensifica en los últimos 6 meses.

### Objetivo
Convertir todo lo aprendido en algo que el mercado vea y valore. Sin esto, los 9 pilares anteriores son un hobby caro.

### Por qué dura
La habilidad de posicionarse, escribir, mostrar trabajo, contactar gente y construir presencia profesional no envejece. Si algo, en una era de IA generativa, gente con presencia auténtica vale más, no menos.

### Sub-pilares

#### A) Vertical: tu nicho

No seas "AI engineer" genérico. Sé el tipo que sabe de **IA aplicada a operaciones de HR/finanzas en empresas medianas**. Específico, vendible, y construye sobre tu experiencia real en BUK y procesos de payroll.

Otras opciones que podrías considerar según tu interés: IA para legal/compliance (encaja con Chile, hay demanda), IA para retail/restaurantes (ya conoces el sector), IA para educación.

Elige uno antes del Pilar 7 y lleva todos los proyectos de ese pilar hacia tu vertical.

#### B) Portfolio: 3–5 proyectos reales

Calidad sobre cantidad. Tu GitHub debe tener al final del plan:
- 1 automatización útil (Pilar 2).
- 1 análisis de datos serio (Pilar 4).
- 1 modelo de ML clásico (Pilar 5).
- 1 modelo de DL desde cero, tipo nano-GPT (Pilar 6).
- 1 proyecto bandera: RAG/agente en producción con tests, deploy, eval (Pilares 7–8).

Cada uno con README excelente. El README es el 50% del valor del proyecto.

#### C) Comunicación

- **Escribe**. Empieza un blog (Substack o Hashnode, gratis). Escribe 1 vez al mes sobre algo que aprendiste. Audiencia 0 al principio, da igual. La práctica de explicar consolida lo aprendido.
- **LinkedIn**: rewrite del perfil entero al final del Pilar 5. Posiciónate explícitamente como Ingeniero Comercial con foco en IA aplicada.
- **GitHub**: README del perfil bien hecho.

#### D) Red

- Únete a 1–2 comunidades activas. Sugerencias: comunidades de Python Chile, slack de Latinoamérica AI, Discord de Anthropic.
- En el último año del plan, asiste a 1 meetup presencial al mes (Santiago tiene varios).
- Habla en uno antes de terminar el plan. Lightning talk de 10 minutos cuenta.

#### E) Open source

A partir del mes 12 del plan, haz al menos 3 contribuciones reales a proyectos open source. Pueden ser pequeñas: documentación, un bug fix, un ejemplo. Un PR aceptado vale más en tu CV que tres certificados.

---

## Hábitos transversales

Cosas que corren en paralelo a todos los pilares.

### Inglés

Ya está en tu radar por Australia, pero también es la lengua de la IA: el 99% de los papers, documentación y debates están en inglés. Tu plan IELTS debería estar alineado con esto, no compitiendo.

Práctica diaria mínima: 15 minutos de input (podcast técnico, video, paper) + 1 vez por semana writing (resumir lo aprendido en inglés).

Podcasts recomendados gratis:
- *Latent Space* (Swyx & Alessio): IA aplicada.
- *The Gradient*: más académico.
- *Lex Fridman* episodios de IA puntuales.

### Lectura

1 libro técnico no urgente cada 2 meses. Sugerencias para los 24 meses:
- *The Pragmatic Programmer* (Hunt & Thomas).
- *Designing Data-Intensive Applications* (Kleppmann). El libro técnico más importante de la década.
- *Hands-On Machine Learning* (Géron).
- *Storytelling with Data* (Knaflic).
- *AI Engineering* (Chip Huyen).
- *The Pragmatic Engineer Newsletter* (Gergely Orosz, suscripción opcional).

### Escritura

1 entrada de blog o post de LinkedIn cada 3 semanas. Sin excepciones. Escribir es pensar. La gente que escribe tiene mejor carrera. Es la palanca más sub-utilizada.

### Construir > consumir

Regla dura: por cada hora de tutorial/video/libro, **una hora de construir tu propia cosa**. No 50/50 nominal; 50/50 real. Si en una semana viste 6 horas de video, construye 6 horas. Si no, frena los videos.

### Salud

No es decoración del plan. 18–24 meses de estudio sostenido se hacen con cuerpo cuidado. Fitness ya está en tu vida; mantenlo no negociable. Sueño antes que cualquier otra cosa. Sin sueño no hay aprendizaje (literalmente, la consolidación de memoria ocurre dormido).

---

## Herramientas: qué pagar, qué no

| Herramienta | Costo | Recomendación |
|---|---|---|
| VS Code | Gratis | Sí, desde día 1. |
| Cursor | ~$20 USD/mes | Recomendable desde el Pilar 2. No antes. |
| Claude Pro | Ya pagado | Mantén. Es tu asistente de aprendizaje. |
| GitHub | Gratis | Sí, esencial. |
| GitHub Copilot | ~$10 USD/mes | Opcional si ya tienes Cursor. Redundante. |
| Coursera Plus | ~$50 USD/mes | No. Audita los cursos gratis. |
| Kaggle | Gratis | Sí, especialmente Kaggle Learn. |
| Hugging Face | Gratis tier suficiente | Sí. |
| Railway/Render | Tier gratis | Sí. |
| Notion / Obsidian | Gratis | Para apuntes. Cualquiera de los dos. |
| ChatGPT Plus | $20 USD/mes | No imprescindible si tienes Claude Pro. A veces útil para contrastar respuestas. |

**Total mínimo viable**: ya estás cubierto. Todo lo esencial es gratis o ya lo tienes.

**Total recomendado**: Cursor ($20) cuando llegues al Pilar 2. Total mensual: ~$20 USD.

---

## Hitos cada 6 meses

Para chequear que vas en buena dirección. Si en un check no llegaste, no es drama — ajustas el ritmo, no la meta.

### Mes 6
- Pilares 0, 1, 2 completos.
- Python: cómodo escribiendo scripts de 100+ líneas sin tutorial.
- Git: usado a diario, sin pánico.
- 2 proyectos personales en GitHub.

### Mes 12
- Pilares 3, 4, 5 completos.
- Puedes entrenar un modelo de ML clásico end-to-end.
- Tienes 1 notebook serio en Kaggle.
- Has empezado el blog.
- LinkedIn parcialmente actualizado.
- Inglés: lectura técnica fluida.

### Mes 18
- Pilares 6, 7 completos.
- Atlas Legal (o tu proyecto bandera equivalente) funcionando.
- Has construido un nano-GPT desde cero.
- Has hecho al menos 1 PR aceptado en open source.
- 3 entradas de blog publicadas mínimo.

### Mes 24
- Pilares 8, 9 completos.
- Atlas Legal desplegado con URL pública.
- CV reposicionado, LinkedIn cerrado.
- Has aplicado a 5+ posiciones técnicas o iniciado conversaciones sobre tu propia empresa.
- Hablaste en 1 meetup como mínimo.

---

## Cómo medir progreso real

Señales auténticas vs ruido:

**Señales reales (úsalas)**
- ¿Puedo construir cosas que antes no podía construir?
- ¿Mi GitHub tiene proyectos que yo mismo usaría?
- ¿Alguien externo a mí me ha pedido ayuda técnica?
- ¿Puedo leer documentación oficial sin frustrarme?
- ¿Puedo explicar lo que aprendí a alguien que no sabe?

**Señales vanidosas (ignóralas)**
- Cantidad de cursos completados.
- Cantidad de certificados.
- Horas de video vistas.
- Estrellas en mi GitHub.
- Followers en LinkedIn.

Una entrevista técnica donde te va bien vale más que diez certificados. Un PR aceptado vale más que cien horas de tutoriales.

---

## Errores que cuestan meses

Trampas frecuentes. Evítalas conscientemente.

1. **Tutorial hell**. Ver tutoriales sin construir. Sensación de progreso, cero progreso real. Antídoto: regla 50/50 ya mencionada.
2. **Optimización prematura del entorno**. Pasar 2 semanas configurando vim/Neovim/tmux antes de saber Python. Tu IDE es un medio, no el objetivo.
3. **Saltar fundamentos por urgencia**. "Quiero hacer agentes ya". Sin Python sólido y sin estadística, los agentes son magia frágil. Confía en el orden de los pilares.
4. **Coleccionar cursos**. Comprar/inscribirse en cursos como sustituto de hacer. El curso que sirve es el que terminas.
5. **Esperar a estar listo para publicar**. Tu primer post va a ser malo. El décimo será decente. Solo se llega al décimo escribiendo los primeros nueve.
6. **Aprender el último framework de moda en lugar del concepto**. LangChain hoy, otra cosa en 18 meses. El concepto de orquestación de LLMs dura. Aprende eso.
7. **Ignorar el lado producto/negocio**. Es tu ventaja. No la diluyas por inseguridad técnica.
8. **Comparación con perfiles de Twitter**. Las personas que se ven brillantes en X llevan años haciéndolo. Estás empezando. Tu comparación es tu yo de hace 6 meses, no un ML engineer de Anthropic.
9. **Estudiar sin descansar**. Burnout a los 9 meses. El plan asume ritmo sostenible. 8–10 horas semanales reales sostenidas vencen a 30 horas semanales por 3 meses.
10. **No documentar lo que aprendes**. Si no lo escribes, se olvida. `polaris-notes` es tu seguro.

---

## Biblioteca permanente

Recursos para consultar y leer durante años, no solo durante el plan.

### Blogs y newsletters
- Chip Huyen (huyenchip.com)
- Eugene Yan (eugeneyan.com)
- Simon Willison (simonwillison.net)
- Andrej Karpathy (karpathy.github.io)
- Latent Space (latent.space)
- The Pragmatic Engineer (Gergely Orosz)
- Anthropic Engineering Blog
- Hugging Face Blog

### Canales de YouTube
- 3Blue1Brown
- StatQuest
- Andrej Karpathy
- Two Minute Papers
- Yannic Kilcher
- Computerphile

### Papers seminales para leer en algún momento de los 2 años
- *Attention Is All You Need* (Vaswani et al, 2017).
- *Language Models are Few-Shot Learners* (Brown et al, 2020 — GPT-3 paper).
- *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al, 2020 — RAG original).
- *Constitutional AI* (Bai et al, 2022 — Anthropic).
- *Chain-of-Thought Prompting Elicits Reasoning in LLMs* (Wei et al, 2022).

### Libros núcleo
- *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow* (Géron).
- *Designing Data-Intensive Applications* (Kleppmann).
- *The Pragmatic Programmer* (Hunt & Thomas).
- *AI Engineering* (Chip Huyen).
- *Practical Statistics for Data Scientists* (Bruce, Bruce, Gedeck).
- *Deep Learning* (Goodfellow, Bengio, Courville — gratis online).

### Cursos referencia
- CS50 (Harvard)
- Machine Learning Specialization (Andrew Ng)
- fast.ai Practical Deep Learning
- Hugging Face NLP Course
- Anthropic Academy

---

## Renovación del plan

Cada **6 meses**, sentarte una hora a revisar:

1. ¿Qué del plan original sigue teniendo sentido?
2. ¿Qué del mundo cambió que el plan no contempla?
3. ¿Qué pilar lleva más tiempo del estimado y por qué?
4. ¿Sigue siendo el vertical correcto?
5. ¿El proyecto bandera sigue siendo el correcto?

El plan es una guía, no una ley. La filosofía (Pilar 0) **no se renegocia**. El orden de pilares puede ajustarse si el contexto cambia mucho. Los recursos específicos pueden reemplazarse si algo mejor aparece.

Si en algún check de 6 meses sentís que perdiste el norte, releé la sección 0 (Filosofía) y la 1 (Ventaja injusta). Esas dos páginas son el plan en miniatura.

---

## Cierre

No es un plan corto. No es un plan fácil. Es un plan **honesto**: lo que aprenderías si fueras alumno mío y yo fuera responsable de que tengas carrera en 2030.

Hay caminos más rápidos para "saber IA" superficialmente. Hay bootcamps de 12 semanas. Hay influencers que prometen empleo en 6 meses. Lo que no hay es atajos para construir el tipo de competencia que sigue valiendo cuando el siguiente hype reemplace al actual. Eso lleva tiempo. Polaris asume que lo tienes.

El primer día, mes 1, semana 1: terminar el Pilar 0. Configurar el entorno. Crear el repo `polaris-notes`. Hacer el primer commit.

El resto se construye sobre eso.

---

*Polaris v1.0 — última revisión: mayo 2026. Próxima revisión sugerida: noviembre 2026.*
