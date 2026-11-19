"""Calculadora principal: evaluación exacta + decimal con alta precisión, complejos y pasos."""
import sympy as sp
from .errors import CalcError
from .fmt import L, approx_text, block, num_text, step
from .parser import parse, parse_unevaluated, split_relation
from . import steps as st


def _clamp_digits(d):
    return max(5, min(int(d), 100))


def _bad(e):
    return e.has(sp.zoo) or e.has(sp.nan)


def evaluate(expr, angle="rad", digits=15, ans=None, want_steps=True):
    digits = _clamp_digits(digits)
    rel = split_relation(expr)
    if rel is not None:
        l, op, r = rel
        a, b = parse(l, angle), parse(r, angle)
        ops = {"=": sp.Eq, "==": sp.Eq, "<": sp.Lt, ">": sp.Gt, "<=": sp.Le, ">=": sp.Ge, "!=": sp.Ne}
        val = ops[op](a, b)
        if val in (sp.true, sp.false):
            return dict(blocks=[block("Resultado", text="Verdadero" if val == sp.true else "Falso")], steps=[], ans=None)
        return dict(blocks=[block("Relación", latex=L(val))], steps=[], ans=None)

    extra = {}
    if ans:
        try:
            extra["ans"] = parse(ans, "rad")
        except CalcError:
            pass
    e = parse(expr, angle, extra)
    if _bad(e):
        raise CalcError("Operación indefinida (por ejemplo, división por cero).")
    r = e
    if not r.free_symbols:
        r = sp.simplify(r)
        if r.has(sp.I):
            r = sp.simplify(sp.expand_complex(r))
    else:
        r = sp.simplify(r)
    if _bad(r):
        raise CalcError("Operación indefinida (por ejemplo, división por cero).")

    blocks = []
    ap = approx_text(r, digits)
    is_plain_int = r.is_Integer
    blocks.append(block("Resultado exacto" if not is_plain_int else "Resultado", latex=L(r)))
    if ap is not None and not (is_plain_int and len(ap) == len(str(r))):
        blocks.append(block(f"Aproximación ({digits} cifras)", text=ap))
    if r.is_number and not r.is_real and r.is_finite is not False and r != sp.zoo:
        mod = sp.simplify(sp.Abs(r))
        phi = sp.arg(r)
        unit = sp.pi / 180 if angle == "deg" else 1
        ang = sp.simplify(phi / unit)
        blocks.append(block("Módulo |z|", latex=L(mod), text=approx_text(mod, digits)))
        blocks.append(block("Argumento" + (" (grados)" if angle == "deg" else " (rad)"), latex=L(ang), text=approx_text(ang, digits)))
        blocks.append(block("Conjugado", latex=L(sp.conjugate(r))))
    if r.free_symbols:
        blocks.append(block("Variables libres", text=", ".join(sorted(str(s) for s in r.free_symbols))))

    steps = []
    if want_steps and not e.free_symbols:
        stages = st.eval_stages(parse_unevaluated(expr, angle))
        if len(stages) > 1:
            steps.append(step("Expresión original", text=expr.strip()))
            for k, s_ in enumerate(stages[1:], 1):
                steps.append(step(f"Paso {k}", L(s_)))
            if L(stages[-1]) != L(r):
                steps.append(step("Resultado simplificado", L(r)))
    return dict(blocks=blocks, steps=steps, ans=str(r) if len(str(r)) < 300 else None)


def to_text(expr, angle="rad", digits=15):
    """Ayuda para otros módulos: valor decimal de una expresión."""
    e = parse(expr, angle)
    return num_text(e, _clamp_digits(digits)) if e.is_real else approx_text(e, digits)
