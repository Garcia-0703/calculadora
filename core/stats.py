"""Estadística descriptiva y regresiones."""
import math
import re
import numpy as np
from .errors import CalcError
from .parser import parse


def parse_numbers(text, name="datos", minimum=1, maximum=5000):
    toks = [t for t in re.split(r"[\s,;]+", text or "") if t]
    if len(toks) < minimum:
        raise CalcError(f"Escribe al menos {minimum} valor(es) en {name}.")
    if len(toks) > maximum:
        raise CalcError(f"Máximo {maximum} valores.")
    out = []
    for t in toks:
        try:
            out.append(float(t))
        except ValueError:
            try:
                v = complex(parse(t).evalf())
                if abs(v.imag) > 1e-12:
                    raise ValueError
                out.append(v.real)
            except Exception:
                raise CalcError(f"«{t}» no es un número válido.")
    arr = np.array(out, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise CalcError("Los datos deben ser números finitos.")
    return arr


def _g(x):
    return float(f"{x:.12g}")


def descriptive(text):
    x = parse_numbers(text)
    n = len(x)
    s = np.sort(x)
    vals, counts = np.unique(x, return_counts=True)
    mode = vals[counts == counts.max()].tolist() if counts.max() > 1 else []
    q1, med, q3 = (float(np.percentile(x, p)) for p in (25, 50, 75))
    var_p = float(np.var(x))
    var_s = float(np.var(x, ddof=1)) if n > 1 else None
    m = float(np.mean(x))
    skew = float(np.mean(((x - m) / np.std(x)) ** 3)) if np.std(x) > 0 else None
    kurt = float(np.mean(((x - m) / np.std(x)) ** 4) - 3) if np.std(x) > 0 else None
    stats = [
        ("n (cantidad)", n), ("Suma", _g(float(np.sum(x)))), ("Media", _g(m)), ("Mediana", _g(med)),
        ("Moda", ", ".join(str(_g(v)) for v in mode) if mode else "no hay"), ("Mínimo", _g(float(s[0]))), ("Máximo", _g(float(s[-1]))),
        ("Rango", _g(float(s[-1] - s[0]))), ("Q1", _g(q1)), ("Q3", _g(q3)), ("Rango intercuartílico", _g(q3 - q1)),
        ("Varianza poblacional σ²", _g(var_p)), ("Desviación estándar poblacional σ", _g(math.sqrt(var_p))),
        ("Varianza muestral s²", _g(var_s) if var_s is not None else "—"),
        ("Desviación estándar muestral s", _g(math.sqrt(var_s)) if var_s is not None else "—"),
        ("Error estándar", _g(math.sqrt(var_s) / math.sqrt(n)) if var_s is not None else "—"),
        ("Coef. de variación", _g(math.sqrt(var_p) / m) if m != 0 else "—"),
        ("Asimetría", _g(skew) if skew is not None else "—"), ("Curtosis (exceso)", _g(kurt) if kurt is not None else "—"),
    ]
    return dict(stats=[dict(name=k, value=v) for k, v in stats], data=x.tolist())


MODELS = {"linear": "Lineal  y = a + b·x", "quadratic": "Cuadrática  y = a + b·x + c·x²", "cubic": "Cúbica",
          "poly": "Polinómica de grado n", "exp": "Exponencial  y = a·e^(b·x)", "log": "Logarítmica  y = a + b·ln x",
          "power": "Potencia  y = a·x^b"}


def regression(xs, ys, model="linear", degree=2, x0=None):
    x = parse_numbers(xs, "X", 2)
    y = parse_numbers(ys, "Y", 2)
    if len(x) != len(y):
        raise CalcError(f"X e Y deben tener la misma cantidad de valores (X: {len(x)}, Y: {len(y)}).")
    n = len(x)
    if model in ("linear", "quadratic", "cubic", "poly"):
        deg = {"linear": 1, "quadratic": 2, "cubic": 3}.get(model, int(degree))
        if not 1 <= deg <= 8:
            raise CalcError("El grado debe estar entre 1 y 8.")
        if n <= deg:
            raise CalcError(f"Para grado {deg} se necesitan al menos {deg + 1} puntos.")
        if len(np.unique(x)) <= deg:
            raise CalcError("Hay pocos valores distintos de X para ese grado.")
        coef = np.polyfit(x, y, deg)[::-1]            # a0, a1, ...
        coef = np.where(np.abs(coef) < 1e-9 * max(np.max(np.abs(coef)), 1e-300), 0.0, coef)
        f = lambda t: sum(c * np.power(t, k) for k, c in enumerate(coef))
        names = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
        eq = "y = " + (" + ".join(f"{c:.6g}" + ("" if k == 0 else ("·x" if k == 1 else f"·x^{k}")) for k, c in enumerate(coef) if c != 0) or "0")
        params = {names[k]: float(c) for k, c in enumerate(coef)}
    elif model == "exp":
        if np.any(y <= 0):
            raise CalcError("La regresión exponencial requiere Y > 0.")
        b, ln_a = np.polyfit(x, np.log(y), 1)
        a = math.exp(ln_a)
        f = lambda t: a * np.exp(b * t)
        eq, params = f"y = {a:.6g}·e^({b:.6g}·x)", {"a": a, "b": float(b)}
    elif model == "log":
        if np.any(x <= 0):
            raise CalcError("La regresión logarítmica requiere X > 0.")
        b, a = np.polyfit(np.log(x), y, 1)
        f = lambda t: a + b * np.log(t)
        eq, params = f"y = {a:.6g} + {b:.6g}·ln(x)", {"a": float(a), "b": float(b)}
    elif model == "power":
        if np.any(x <= 0) or np.any(y <= 0):
            raise CalcError("La regresión de potencia requiere X > 0 y Y > 0.")
        b, ln_a = np.polyfit(np.log(x), np.log(y), 1)
        a = math.exp(ln_a)
        f = lambda t: a * np.power(t, b)
        eq, params = f"y = {a:.6g}·x^{b:.6g}", {"a": a, "b": float(b)}
    else:
        raise CalcError("Modelo de regresión no soportado.")
    yhat = f(x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    out = dict(model=model, equation=eq.replace("--", "+ ").replace("+ -", "- "), params=params, r2=r2,
               r=(float(np.corrcoef(x, y)[0, 1]) if model == "linear" and np.std(x) > 0 and np.std(y) > 0 else None),
               rmse=math.sqrt(ss_res / n), n=n, x=x.tolist(), y=y.tolist())
    lo, hi = float(x.min()), float(x.max())
    pad = (hi - lo) * 0.05 or 1.0
    xx = np.linspace(max(lo - pad, 1e-9) if model in ("log", "power") else lo - pad, hi + pad, 200)
    with np.errstate(all="ignore"):
        yy = f(xx)
    out["curve"] = dict(x=xx.tolist(), y=[float(v) if np.isfinite(v) else None for v in yy])
    if x0 not in (None, ""):
        try:
            p = float(x0)
        except (TypeError, ValueError):
            raise CalcError("El valor a predecir debe ser un número.")
        with np.errstate(all="ignore"):
            pv = float(f(np.array([p]))[0])
        out["prediction"] = dict(x=p, y=pv if math.isfinite(pv) else None)
    return out
