"""Álgebra simbólica: simplificar, expandir, factorizar, fracciones parciales, sustitución..."""
import sympy as sp
from .errors import CalcError
from .fmt import L, approx_text, block, step
from .parser import parse

OPS = {
    "simplify": ("Simplificar", sp.simplify), "expand": ("Expandir", sp.expand), "factor": ("Factorizar", sp.factor),
    "cancel": ("Cancelar factores comunes", sp.cancel), "apart": ("Fracciones parciales", sp.apart),
    "together": ("Combinar en una fracción", sp.together), "trigsimp": ("Simplificar trigonométrica", sp.trigsimp),
    "expand_trig": ("Expandir trigonométrica", sp.expand_trig), "expand_log": ("Expandir logaritmos", lambda e: sp.expand_log(e, force=True)),
    "logcombine": ("Combinar logaritmos", lambda e: sp.logcombine(e, force=True)), "radsimp": ("Racionalizar", sp.radsimp),
}


def run(expr, op="simplify", var=None, value=None, digits=15):
    e = parse(expr)
    blocks, steps = [], [step("Expresión original", L(e))]
    if op == "subs":
        if not var or value is None:
            raise CalcError("Indica la variable y el valor a sustituir.")
        v = sp.Symbol(var.strip())
        val = parse(str(value))
        r = sp.simplify(e.subs(v, val))
        steps.append(step(f"Sustituir {var} = {value}", L(e.subs(v, val, simultaneous=True))))
        label = "Valor numérico"
    elif op == "collect":
        if not var:
            raise CalcError("Indica la variable para agrupar.")
        r = sp.collect(sp.expand(e), sp.Symbol(var.strip()))
        label = "Agrupado"
    elif op in OPS:
        label, fn = OPS[op]
        try:
            r = fn(e)
        except Exception:
            raise CalcError("No se pudo aplicar esa operación a la expresión.")
        steps.append(step(label, L(r)))
    else:
        raise CalcError("Operación no soportada.")
    blocks.append(block(label if op != "simplify" else "Resultado simplificado", latex=L(r)))
    ap = approx_text(r, digits)
    if ap is not None and not r.is_Integer:
        blocks.append(block("Aproximación", text=ap))
    if op == "factor" and e != r:
        blocks.append(block("Igualdad", latex=f"{L(e)} = {L(r)}"))
    return dict(blocks=blocks, steps=steps if len(steps) > 1 else [])
