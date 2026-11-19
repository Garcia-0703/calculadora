# ∑ Calculadora científica web

Calculadora científica de alta precisión con **backend FastAPI + sympy** (cálculos) y **frontend web** (KaTeX + Plotly.js).
Pensada para desplegarse en **Vercel**.

## Funciones
| Herramienta | Qué hace |
|---|---|
| **Calculadora** | Resultado exacto y decimal (hasta 100 cifras), números complejos (módulo, argumento, conjugado), factoriales, grados/radianes, `Ans`, historial y **paso a paso** |
| **Álgebra** | Simplificar, expandir, factorizar, fracciones parciales, trigonometría, logaritmos, racionalizar, agrupar, sustituir |
| **Ecuaciones** | Ecuaciones (lineal, cuadrática con discriminante, polinómicas, trascendentes con raíces numéricas), sistemas (lineales con Gauss-Jordan y no lineales) e inecuaciones |
| **Derivadas e integrales** | Derivadas con reglas explicadas (suma, producto, cociente, cadena), integrales indefinidas/definidas/impropias con pasos, límites, series de Taylor y gradiente |
| **Matrices** | Determinante, inversa, rango, RREF, autovalores/autovectores, núcleo, A±B, A·B, Aⁿ, A·X=B (con pasos de Gauss-Jordan) |
| **Estadística** | 19 medidas descriptivas, histograma y caja; regresión lineal, polinómica, exponencial, logarítmica y de potencia con R², predicción y gráfica |
| **Gráficas** | Hasta 5 funciones 2D (corta asíntotas) y superficies 3D z = f(x, y) interactivas |
| **Unidades y constantes** | 14 categorías de conversión (incl. temperatura) y 32 constantes físicas/matemáticas |

Sintaxis: `2x`, `3(x+1)`, `x^2`, `5!`, `sqrt(2)`/`√2`, `30°`, `pi`/`π`, `e`, `i`, `ln`, `log` (base 10), `log2`, `abs`, `sin`…`atanh`, `root(x,n)`, `binomial(n,k)`.
Los decimales se tratan como racionales exactos (`0.1+0.2 = 3/10`).

## Estructura
```
app.py            # API FastAPI (exporta `app`, lo que Vercel busca) — solo validación y rutas
core/             # lógica matemática (sin web): parser seguro, calc, álgebra, solver, cálculo, matrices, stats, gráficas, unidades
public/index.html # frontend (se sirve desde la CDN de Vercel)
tests/            # pruebas del núcleo
```
Endpoints: `POST /api/{calc,algebra,solve,calculus,matrix,stats,regression,graph2d,graph3d,units/convert}`,
`GET /api/{units/catalog,constants,health}`, documentación en `/api/docs`.
Los errores de entrada devuelven **HTTP 422** con `{"error": {"message": ...}}`; la interfaz los muestra sin romperse.

## Seguridad
El texto del usuario **nunca se pasa a `eval` directamente**: se valida token a token (lista blanca de funciones y operadores; sin atributos,
cadenas ni guiones bajos), se estima el tamaño del resultado antes de calcular (bloquea `9**9**9`, `100000!`…) y se analiza con `__builtins__` vacío.
Límites: 400 caracteres por expresión, matrices ≤ 8×8, factorial/gamma ≤ 3000.

## Ejecutar en local
```bash
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app:app --reload                                  # http://127.0.0.1:8000
python -m unittest discover -s tests                      # pruebas (opcional)
```

## Desplegar en Vercel
1. Sube el contenido de esta carpeta a un repositorio de GitHub (`app.py`, `requirements.txt`, `core/` y `public/` en la **raíz**).
2. vercel.com → *Add New → Project* → importa el repositorio → **Deploy** (Vercel detecta FastAPI por la variable `app`).
3. Queda en `https://TU-PROYECTO.vercel.app`. La primera petición tras un rato inactivo puede tardar unos segundos (arranque de sympy).
