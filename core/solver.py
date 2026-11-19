"""Ecuaciones, sistemas e inecuaciones."""
import re
import numpy as np
import sympy as sp
from .errors import CalcError
from .fmt import L, approx_text, block, step
from .parser import parse_relation
from . import steps as st

_MAXV = 6


def _pick_var(symbols, var):
    if var:
        return sp.Symbol(var.strip())
    if not symbols:
        return None
    for s in symbols:
        if s.name == "x":
            return s
    return sorted(symbols, key=lambda s: s.name)[0]


def numeric_roots(f, x, lo=-50, hi=50, n=4001):
    """Raíces reales por cambio de signo + bisección (para ecuaciones sin solución exacta)."""
    try:
        fn = sp.lambdify(x, f, "numpy")
        xs = np.linspace(lo, hi, n)
        with np.errstate(all="ignore"):
            ys = np.real_if_close(fn(xs))
        ys = np.asarray(ys, dtype=float)
    except Exception:
        return []
    roots = []
    for i in range(len(xs) - 1):
        a, b = ys[i], ys[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0:
            roots.append(float(xs[i]))
        elif a * b < 0:
            lo_, hi_ = xs[i], xs[i + 1]
            for _ in range(80):
                mid = (lo_ + hi_) / 2
                with np.errstate(all="ignore"):
                    fm = float(np.real(fn(mid)))
                if fm == 0 or not np.isfinite(fm):
                    break
                if np.sign(fm) == np.sign(float(np.real(fn(lo_)))):
                    lo_ = mid
                else:
                    hi_ = mid
            r = (lo_ + hi_) / 2
            with np.errstate(all="ignore"):
                if abs(float(np.real(fn(r)))) < 1e-6 * max(1, abs(r)):
                    roots.append(r)
    out = []
    for r in roots:
        if not any(abs(r - o) < 1e-7 for o in out):
            out.append(r)
    return out


def _solutions_block(sols, x, digits):
    blocks = []
    for k, s in enumerate(sols, 1):
        lab = f"{L(x)}_{{{k}}}" if len(sols) > 1 else L(x)
        blocks.append(block(f"Solución {k}" if len(sols) > 1 else "Solución", latex=f"{lab} = {L(s)}", text=approx_text(s, digits) if s.is_number else None))
    return blocks


def solve_equation(lhs, rhs, var, digits):
    f = sp.expand(lhs - rhs)
    syms = sorted(f.free_symbols, key=lambda s: s.name)
    if not syms:
        ok = sp.simplify(f) == 0
        return dict(blocks=[block("Resultado", text="Identidad verdadera (todos los valores cumplen)" if ok else "No tiene solución (contradicción)")], steps=[])
    x = _pick_var(syms, var)
    if x not in syms:
        raise CalcError(f"La variable «{x}» no aparece en la ecuación.")
    steps = st.solve_steps(f, x)
    blocks = [block("Ecuación", latex=f"{L(lhs)} = {L(rhs)}")]
    others = [s for s in syms if s != x]
    if others:
        blocks.append(block("Incógnita", text=f"Se resuelve para {x} (parámetros: {', '.join(map(str, others))})"))
    try:
        if f.is_polynomial(x):
            sols = sp.solve(f, x)
            sols = [sp.simplify(s) for s in sols]
            if not sols:
                blocks.append(block("Resultado", text="Sin solución"))
            else:
                blocks += _solutions_block(sols, x, digits)
        else:
            ss = sp.solveset(f, x, sp.S.Reals)
            if isinstance(ss, sp.FiniteSet):
                blocks += _solutions_block(list(ss), x, digits) if len(ss) else [block("Resultado", text="Sin solución real")]
            elif isinstance(ss, sp.ConditionSet) or ss.has(sp.ConditionSet):
                rs = numeric_roots(f, x) if not others else []
                if rs:
                    blocks.append(block("Raíces numéricas (en [−50, 50])", text="; ".join(f"{x} ≈ {r:.10g}" for r in rs[:20])))
                    steps.append(step("Búsqueda numérica", text="No hay fórmula cerrada; se localizaron las raíces por cambio de signo y bisección."))
                else:
                    blocks.append(block("Resultado", text="No encontré soluciones reales"))
            else:
                blocks.append(block("Conjunto solución", latex=L(ss)))
    except CalcError:
        raise
    except Exception:
        raise CalcError("No pude resolver esta ecuación.")
    return dict(blocks=blocks, steps=steps)


def solve_inequality(lhs, op, rhs, var, digits):
    f = lhs - rhs
    syms = sorted(f.free_symbols, key=lambda s: s.name)
    if len(syms) != 1:
        raise CalcError("Las inecuaciones se resuelven con una sola variable.")
    x = syms[0]
    rel = {"<": sp.Lt, ">": sp.Gt, "<=": sp.Le, ">=": sp.Ge}.get(op)
    if rel is None:
        raise CalcError("Operador de inecuación no soportado.")
    try:
        xr = sp.Symbol(x.name, real=True)
        sol = sp.solve_univariate_inequality(rel(f.subs(x, xr), 0), xr, relational=False)
    except Exception:
        raise CalcError("No pude resolver esta inecuación.")
    steps = [step("Llevar todo a un lado", rf"{L(sp.expand(f))}\ {L_op(op)}\ 0")]
    try:
        crit = sp.solve(sp.numer(sp.together(f)), x)
        crit = sorted({c for c in crit if c.is_real}, key=lambda c: float(c))
        if crit:
            steps.append(step("Puntos críticos (ceros del numerador)", r",\ ".join(f"{L(x)}={L(c)}" for c in crit)))
        den = sp.denom(sp.together(f))
        if den.has(x):
            dz = sorted({c for c in sp.solve(den, x) if c.is_real}, key=lambda c: float(c))
            if dz:
                steps.append(step("Puntos donde se anula el denominador (excluidos)", r",\ ".join(f"{L(x)}={L(c)}" for c in dz)))
        steps.append(step("Estudiar el signo en cada intervalo", text="Se analiza el signo de la expresión entre los puntos críticos."))
    except Exception:
        pass
    blocks = [block("Inecuación", latex=f"{L(lhs)}\\ {L_op(op)}\\ {L(rhs)}"), block("Solución (intervalos)", latex=L(sol))]
    return dict(blocks=blocks, steps=steps)


def L_op(op):
    return {"<": "<", ">": ">", "<=": r"\le", ">=": r"\ge", "=": "=", "!=": r"\ne"}[op]


def solve_system(rels, var, digits):
    eqs, syms = [], set()
    for l, op, r in rels:
        if op != "=":
            raise CalcError("En un sistema todas las líneas deben ser ecuaciones (=).")
        eqs.append(sp.expand(l - r))
        syms |= (l - r).free_symbols
    syms = sorted(syms, key=lambda s: s.name)
    if not syms:
        raise CalcError("El sistema no tiene incógnitas.")
    if len(syms) > _MAXV:
        raise CalcError(f"Máximo {_MAXV} incógnitas.")
    blocks = [block("Sistema", latex=r"\begin{cases}" + r"\\ ".join(f"{L(l)}={L(r)}" for l, _, r in rels) + r"\end{cases}")]
    steps = []
    linear = all(sp.Poly(e, *syms).total_degree() <= 1 for e in eqs)
    try:
        if linear:
            A, b = sp.linear_eq_to_matrix(eqs, syms)
            _, steps = st.gauss_jordan(A.row_join(b), ncoef=len(syms))
            sol = sp.linsolve((A, b), *syms)
            if sol == sp.EmptySet:
                blocks.append(block("Resultado", text="Sistema incompatible (sin solución)"))
            else:
                tup = list(sol)[0]
                free = set().union(*[t.free_symbols for t in tup]) & set(syms)
                for s_, v in zip(syms, tup):
                    blocks.append(block(f"{s_}", latex=f"{L(s_)} = {L(v)}", text=approx_text(v, digits) if v.is_number else None))
                if free:
                    blocks.append(block("Nota", text="Sistema indeterminado: infinitas soluciones (dependen de parámetros libres)."))
        else:
            sols = sp.solve(eqs, syms, dict=True)
            if not sols:
                blocks.append(block("Resultado", text="No encontré soluciones"))
            for k, d in enumerate(sols[:12], 1):
                blocks.append(block(f"Solución {k}", latex=r",\ ".join(f"{L(k_)}={L(v)}" for k_, v in d.items())))
    except CalcError:
        raise
    except Exception:
        raise CalcError("No pude resolver este sistema.")
    return dict(blocks=blocks, steps=steps)


def run(text, var=None, digits=15, angle="rad"):
    lines = [l.strip() for l in re.split(r"[\n;]+", text or "") if l.strip()]
    if not lines:
        raise CalcError("Escribe una ecuación, un sistema (una por línea) o una inecuación.")
    if len(lines) > _MAXV:
        raise CalcError(f"Máximo {_MAXV} ecuaciones.")
    rels = []
    for l in lines:
        a, op, b = parse_relation(l, angle)
        rels.append((a, op or "=", b if op else sp.Integer(0)))
    if len(rels) == 1:
        a, op, b = rels[0]
        return solve_equation(a, b, var, digits) if op == "=" else solve_inequality(a, op, b, var, digits)
    return solve_system(rels, var, digits)
