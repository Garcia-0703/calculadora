"""Derivadas, integrales (definidas e indefinidas), límites, series y gradiente."""
import sympy as sp
from .errors import CalcError
from .fmt import L, approx_text, block, step
from .parser import parse
from . import steps as st


def _var(expr, var):
    if var and var.strip():
        return sp.Symbol(var.strip())
    fs = sorted(expr.free_symbols, key=lambda s: s.name)
    if not fs:
        return sp.Symbol("x")
    return next((s for s in fs if s.name == "x"), fs[0])


def derivative(expr, var=None, order=1, at=None, digits=15):
    e = parse(expr)
    x = _var(e, var)
    order = int(order)
    if not 1 <= order <= 6:
        raise CalcError("El orden de la derivada debe estar entre 1 y 6.")
    steps, cur = [], e
    for k in range(1, order + 1):
        s, cur_next = st.derivative_steps(cur, x)
        if order > 1:
            steps.append(step(f"Derivada de orden {k}", rf"\frac{{d^{{{k}}}}}{{d{L(x)}^{{{k}}}}}\left({L(e)}\right)={L(cur_next)}"))
        if k == 1:
            steps = s + steps if order == 1 else s + steps
        cur = cur_next
    blocks = [block("Función", latex=f"f({L(x)}) = {L(e)}"),
              block("Derivada" + ("" if order == 1 else f" de orden {order}"), latex=rf"\frac{{d{'^'+str(order) if order>1 else ''}f}}{{d{L(x)}{'^'+str(order) if order>1 else ''}}} = {L(cur)}")]
    if at not in (None, ""):
        p = parse(str(at))
        v = sp.simplify(cur.subs(x, p))
        blocks.append(block(f"Evaluada en {L(x)} = {at}", latex=L(v), text=approx_text(v, digits)))
    return dict(blocks=blocks, steps=steps)


def gradient(expr, digits=15):
    e = parse(expr)
    syms = sorted(e.free_symbols, key=lambda s: s.name)
    if len(syms) < 2:
        raise CalcError("El gradiente requiere una función de al menos 2 variables (p. ej. x^2 + x*y).")
    g = [sp.simplify(sp.diff(e, s)) for s in syms]
    blocks = [block("Función", latex=L(e))]
    for s, gi in zip(syms, g):
        blocks.append(block(rf"∂f/∂{s}", latex=rf"\frac{{\partial f}}{{\partial {L(s)}}}={L(gi)}"))
    blocks.append(block("Gradiente ∇f", latex=r"\nabla f=\left(" + r",\ ".join(L(gi) for gi in g) + r"\right)"))
    return dict(blocks=blocks, steps=[])


def integral(expr, var=None, a=None, b=None, digits=15):
    e = parse(expr)
    x = _var(e, var)
    definite = a not in (None, "") and b not in (None, "")
    steps = st.integral_steps(e, x)
    try:
        F = sp.integrate(e, x)
    except Exception:
        F = sp.Integral(e, x)
    unresolved = isinstance(F, sp.Integral) or F.has(sp.Integral)
    blocks = [block("Integrando", latex=rf"\int {L(e)}\,d{L(x)}")]
    if not definite:
        if unresolved:
            raise CalcError("No encontré una antiderivada en forma cerrada para esta función.")
        blocks.append(block("Integral indefinida", latex=rf"{L(F)}+C"))
        return dict(blocks=blocks, steps=steps)
    lo, hi = parse(str(a)), parse(str(b))
    head = rf"\int_{{{L(lo)}}}^{{{L(hi)}}} {L(e)}\,d{L(x)}"
    blocks[0] = block("Integral definida", latex=head)
    if not unresolved:
        steps.append(step("Teorema fundamental del cálculo", rf"\int_a^b f\,dx = F(b)-F(a),\quad F={L(F)}"))
        val = sp.limit(F, x, hi, "-") - sp.limit(F, x, lo, "+") if (hi in (sp.oo, -sp.oo) or lo in (sp.oo, -sp.oo)) else F.subs(x, hi) - F.subs(x, lo)
        val = sp.simplify(val)
        steps.append(step("Evaluar F(b) − F(a)", rf"{L(val)}"))
    else:
        try:
            val = sp.Integral(e, (x, lo, hi)).evalf(max(digits, 15))
        except Exception:
            raise CalcError("No pude calcular la integral definida.")
        steps.append(step("Integración numérica", text="No hay antiderivada elemental; se evaluó numéricamente."))
    if val.has(sp.zoo) or val.has(sp.nan):
        raise CalcError("La integral no converge o está indefinida en esos límites.")
    blocks.append(block("Valor exacto", latex=L(val)))
    ap = approx_text(val, digits)
    if ap is not None and not val.is_Integer:
        blocks.append(block("Aproximación", text=ap))
    return dict(blocks=blocks, steps=steps)


def limit(expr, var=None, point="0", direction="both", digits=15):
    e = parse(expr)
    x = _var(e, var)
    p = parse(str(point))
    d = {"both": "+-", "right": "+", "left": "-"}.get(direction, "+-")
    try:
        v = sp.limit(e, x, p, d)
    except Exception:
        raise CalcError("No pude calcular el límite.")
    if isinstance(v, sp.Limit):
        raise CalcError("No pude calcular el límite.")
    sub = {"+-": "", "+": "^{+}", "-": "^{-}"}[d]
    blocks = [block("Límite", latex=rf"\lim_{{{L(x)}\to {L(p)}{sub}}} {L(e)} = {L(v)}")]
    ap = approx_text(v, digits)
    if ap is not None and not v.is_Integer:
        blocks.append(block("Aproximación", text=ap))
    return dict(blocks=blocks, steps=[])


def series(expr, var=None, point="0", n=6):
    e = parse(expr)
    x = _var(e, var)
    n = max(1, min(int(n), 15))
    try:
        s = sp.series(e, x, parse(str(point)), n)
    except Exception:
        raise CalcError("No pude calcular la serie de Taylor.")
    return dict(blocks=[block(f"Serie de Taylor (orden {n})", latex=L(s))], steps=[])


def run(kind, **kw):
    fn = {"derivative": derivative, "integral": integral, "limit": limit, "series": series, "gradient": gradient}.get(kind)
    if fn is None:
        raise CalcError("Operación de cálculo no soportada.")
    return fn(**kw)
