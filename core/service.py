"""Casos de uso del backend: dict de entrada -> dict de salida (sin FastAPI; fácil de probar)."""
import functools
from . import algebra, calc, calculus, graph, matrices, solver, stats, units
from .errors import CalcError


def safe(fn):
    """Cualquier fallo inesperado de sympy/numpy se convierte en un error 422 amigable."""
    @functools.wraps(fn)
    def w(d):
        try:
            return fn(d)
        except CalcError:
            raise
        except RecursionError:
            raise CalcError("La expresión es demasiado compleja para resolverla aquí.")
        except Exception:
            raise CalcError("No pude calcular esto: la expresión es demasiado compleja o no está soportada.")
    return w


@safe
def calc_eval(d):
    return calc.evaluate(d["expr"], d.get("angle", "rad"), d.get("digits", 15), d.get("ans"), d.get("steps", True))


@safe
def algebra_run(d):
    return algebra.run(d["expr"], d.get("op", "simplify"), d.get("var"), d.get("value"), d.get("digits", 15))


@safe
def solve_run(d):
    return solver.run(d["text"], d.get("var"), d.get("digits", 15), d.get("angle", "rad"))


@safe
def calculus_run(d):
    kind = d["kind"]
    kw = {k: v for k, v in d.items() if k != "kind" and v is not None}
    allowed = {"derivative": {"expr", "var", "order", "at", "digits"}, "integral": {"expr", "var", "a", "b", "digits"},
               "limit": {"expr", "var", "point", "direction", "digits"}, "series": {"expr", "var", "point", "n"},
               "gradient": {"expr", "digits"}}.get(kind)
    if allowed is None:
        raise CalcError("Operación de cálculo no soportada.")
    return calculus.run(kind, **{k: v for k, v in kw.items() if k in allowed})


@safe
def matrix_run(d):
    return matrices.run(d["op"], d["A"], d.get("B"), d.get("k"), d.get("digits", 15))


@safe
def stats_describe(d):
    return stats.descriptive(d["data"])


@safe
def stats_regression(d):
    return stats.regression(d["x"], d["y"], d.get("model", "linear"), d.get("degree", 2), d.get("x0"))


@safe
def graph_2d(d):
    return graph.graph2d(d["exprs"], d.get("xmin", -10), d.get("xmax", 10), d.get("n", 800), d.get("angle", "rad"))


@safe
def graph_3d(d):
    return graph.graph3d(d["expr"], d.get("xmin", -5), d.get("xmax", 5), d.get("ymin", -5), d.get("ymax", 5), d.get("n", 50), d.get("angle", "rad"))


@safe
def units_convert(d):
    return units.convert(d["category"], d["value"], d["frm"], d["to"])


def units_catalog():
    return dict(categories=units.catalog())


def constants_list():
    return dict(constants=units.constants())
