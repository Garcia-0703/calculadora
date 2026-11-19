"""Backend FastAPI de la calculadora científica. Vercel detecta la variable `app`.
La matemática vive en core/; el frontend estático está en public/."""
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from core import service
from core.errors import CalcError

app = FastAPI(title="Calculadora científica API", docs_url="/api/docs", openapi_url="/api/openapi.json")

Angle = Literal["rad", "deg"]


class CalcIn(BaseModel):
    expr: str = Field(..., max_length=500)
    angle: Angle = "rad"
    digits: int = Field(15, ge=5, le=100)
    ans: Optional[str] = Field(None, max_length=400)
    steps: bool = True


class AlgebraIn(BaseModel):
    expr: str = Field(..., max_length=500)
    op: str = Field("simplify", max_length=20)
    var: Optional[str] = Field(None, max_length=10)
    value: Optional[str] = Field(None, max_length=100)
    digits: int = Field(15, ge=5, le=100)


class SolveIn(BaseModel):
    text: str = Field(..., max_length=1500)
    var: Optional[str] = Field(None, max_length=10)
    digits: int = Field(15, ge=5, le=100)
    angle: Angle = "rad"


class CalculusIn(BaseModel):
    kind: Literal["derivative", "integral", "limit", "series", "gradient"]
    expr: str = Field(..., max_length=500)
    var: Optional[str] = Field(None, max_length=10)
    order: Optional[int] = Field(None, ge=1, le=6)
    at: Optional[str] = Field(None, max_length=100)
    a: Optional[str] = Field(None, max_length=100)
    b: Optional[str] = Field(None, max_length=100)
    point: Optional[str] = Field(None, max_length=100)
    direction: Optional[Literal["both", "left", "right"]] = None
    n: Optional[int] = Field(None, ge=1, le=15)
    digits: int = Field(15, ge=5, le=100)


class MatrixIn(BaseModel):
    op: str = Field(..., max_length=20)
    A: str = Field(..., max_length=3000)
    B: Optional[str] = Field(None, max_length=3000)
    k: Optional[str] = Field(None, max_length=100)
    digits: int = Field(15, ge=5, le=100)


class StatsIn(BaseModel):
    data: str = Field(..., max_length=40000)


class RegressionIn(BaseModel):
    x: str = Field(..., max_length=40000)
    y: str = Field(..., max_length=40000)
    model: str = Field("linear", max_length=20)
    degree: int = Field(2, ge=1, le=8)
    x0: Optional[str] = Field(None, max_length=50)


class Graph2DIn(BaseModel):
    exprs: List[str] = Field(..., max_length=5)
    xmin: str = Field("-10", max_length=40)
    xmax: str = Field("10", max_length=40)
    n: int = Field(800, ge=50, le=3000)
    angle: Angle = "rad"


class Graph3DIn(BaseModel):
    expr: str = Field(..., max_length=500)
    xmin: str = Field("-5", max_length=40)
    xmax: str = Field("5", max_length=40)
    ymin: str = Field("-5", max_length=40)
    ymax: str = Field("5", max_length=40)
    n: int = Field(50, ge=10, le=80)
    angle: Angle = "rad"


class UnitsIn(BaseModel):
    category: str = Field(..., max_length=30)
    value: str = Field(..., max_length=40)
    frm: str = Field(..., max_length=20)
    to: str = Field(..., max_length=20)


def _err(status, code, msg):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": msg}})


@app.exception_handler(CalcError)
async def calc_error(request: Request, exc: CalcError):
    return _err(exc.http_status, "CalcError", str(exc))


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    where = ".".join(str(x) for x in first.get("loc", [])[1:]) or "entrada"
    return _err(422, "ValidationError", f"Entrada no válida en «{where}»: {first.get('msg', 'valor incorrecto')}.")


@app.exception_handler(Exception)
async def unexpected(request: Request, exc: Exception):
    return _err(500, "InternalError", "Error interno del servidor al calcular. Inténtalo de nuevo.")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/calc")
def calc(body: CalcIn):
    return service.calc_eval(body.model_dump())


@app.post("/api/algebra")
def algebra(body: AlgebraIn):
    return service.algebra_run(body.model_dump())


@app.post("/api/solve")
def solve(body: SolveIn):
    return service.solve_run(body.model_dump())


@app.post("/api/calculus")
def calculus(body: CalculusIn):
    return service.calculus_run(body.model_dump())


@app.post("/api/matrix")
def matrix(body: MatrixIn):
    return service.matrix_run(body.model_dump())


@app.post("/api/stats")
def stats(body: StatsIn):
    return service.stats_describe(body.model_dump())


@app.post("/api/regression")
def regression(body: RegressionIn):
    return service.stats_regression(body.model_dump())


@app.post("/api/graph2d")
def graph2d(body: Graph2DIn):
    return service.graph_2d(body.model_dump())


@app.post("/api/graph3d")
def graph3d(body: Graph3DIn):
    return service.graph_3d(body.model_dump())


@app.post("/api/units/convert")
def units_convert(body: UnitsIn):
    return service.units_convert(body.model_dump())


@app.get("/api/units/catalog")
def units_catalog():
    return service.units_catalog()


@app.get("/api/constants")
def constants():
    return service.constants_list()


@app.get("/", include_in_schema=False)
def index():  # en Vercel public/ lo sirve la CDN; esto cubre el uso local con uvicorn
    f = Path(__file__).parent / "public" / "index.html"
    return FileResponse(f) if f.exists() else JSONResponse({"status": "ok"})
