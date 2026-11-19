"""Muestreo numérico de funciones para graficar (2D: y=f(x), 3D: z=f(x,y))."""
import numpy as np
import sympy as sp
from .errors import CalcError
from .fmt import L
from .parser import parse


def _num(v, name):
    try:
        v = float(parse(str(v)).evalf())
    except Exception:
        raise CalcError(f"{name} debe ser un número.")
    if not np.isfinite(v):
        raise CalcError(f"{name} debe ser finito.")
    return v


def _clean(arr, ymax=1e6):
    a = np.asarray(arr)
    if np.iscomplexobj(a):
        a = np.where(np.abs(a.imag) < 1e-9, a.real, np.nan)
    a = a.astype(float)
    a = np.where(np.abs(a) > ymax, np.nan, a)
    return a


def _lambdify(expr, vars_):
    extra = expr.free_symbols - set(vars_)
    if extra:
        raise CalcError("La función solo puede usar la(s) variable(s): " + ", ".join(map(str, vars_)) + ". Sobra: " + ", ".join(sorted(map(str, extra))))
    try:
        return sp.lambdify(vars_, expr, modules=["numpy"])
    except Exception:
        raise CalcError("No pude convertir la función para graficarla.")


def graph2d(exprs, xmin=-10, xmax=10, n=800, angle="rad"):
    exprs = [e for e in exprs if e and e.strip()]
    if not exprs:
        raise CalcError("Escribe al menos una función, por ejemplo: sin(x)")
    if len(exprs) > 5:
        raise CalcError("Máximo 5 funciones.")
    lo, hi = _num(xmin, "x mín"), _num(xmax, "x máx")
    if not lo < hi:
        raise CalcError("x mín debe ser menor que x máx.")
    n = max(50, min(int(n), 3000))
    x = sp.Symbol("x")
    xs = np.linspace(lo, hi, n)
    series = []
    for t in exprs:
        e = parse(t, angle)
        f = _lambdify(e, [x])
        with np.errstate(all="ignore"):
            try:
                y = f(xs)
            except Exception:
                raise CalcError(f"No pude evaluar «{t}».")
        y = np.full_like(xs, float(y)) if np.ndim(y) == 0 else y
        y = _clean(y)
        # cortar saltos (asíntotas): si hay cambio de signo con magnitud enorme, insertar hueco
        dy = np.abs(np.diff(y))
        fin = y[np.isfinite(y)]
        if fin.size > 2:
            span = max(np.percentile(np.abs(fin), 95), 1.0)
            thr = 4 * max(1.0, float(np.median(np.abs(fin))))
            sign_flip = np.sign(y[:-1]) * np.sign(y[1:]) < 0
            jump = np.where((dy > 20 * span) | (sign_flip & (dy > thr)))[0]
            for j in jump:
                y[j + 1] = np.nan
        series.append(dict(expr=t, latex=L(e), y=[None if not np.isfinite(v) else float(v) for v in y]))
    return dict(x=xs.tolist(), series=series)


def graph3d(expr, xmin=-5, xmax=5, ymin=-5, ymax=5, n=50, angle="rad"):
    if not expr or not expr.strip():
        raise CalcError("Escribe una función de x e y, por ejemplo: sin(x)*cos(y)")
    x0, x1, y0, y1 = _num(xmin, "x mín"), _num(xmax, "x máx"), _num(ymin, "y mín"), _num(ymax, "y máx")
    if not (x0 < x1 and y0 < y1):
        raise CalcError("Los mínimos deben ser menores que los máximos.")
    n = max(10, min(int(n), 80))
    x, y = sp.symbols("x y")
    e = parse(expr, angle)
    f = _lambdify(e, [x, y])
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    X, Y = np.meshgrid(xs, ys)
    with np.errstate(all="ignore"):
        try:
            Z = f(X, Y)
        except Exception:
            raise CalcError("No pude evaluar la función en la malla.")
    Z = np.full_like(X, float(Z)) if np.ndim(Z) == 0 else Z
    Z = _clean(Z, 1e4)
    return dict(x=xs.tolist(), y=ys.tolist(), z=[[None if not np.isfinite(v) else float(v) for v in row] for row in Z], latex=L(e))
