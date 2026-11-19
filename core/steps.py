"""Generadores de 'paso a paso' (derivadas, integrales, ecuaciones, evaluación numérica, Gauss-Jordan)."""
import sympy as sp
from sympy.integrals import manualintegrate as mi

from .fmt import L, step

# ------------------------------------------------------------------ evaluación numérica por etapas
def eval_stages(tree, max_stages=12):
    """tree: expresión SIN evaluar. Devuelve lista de expresiones intermedias (innermost first)."""
    if tree is None:
        return []

    def once(e):
        if e.is_Atom:
            return e
        if all(a.is_Atom for a in e.args):
            try:
                return e.func(*e.args)
            except Exception:
                return e
        return e.func(*[once(a) for a in e.args], evaluate=False)

    out, cur = [tree], tree
    for _ in range(max_stages):
        if cur.is_Atom:
            break
        try:
            nxt = once(cur)
        except Exception:
            break
        if nxt == cur and str(nxt) == str(cur):
            break
        out.append(nxt)
        cur = nxt
    uniq = []
    for e in out:
        if not uniq or L(e) != L(uniq[-1]):
            uniq.append(e)
    return uniq


# ------------------------------------------------------------------ derivadas
_FUNC_RULES = {sp.sin: "Derivada del seno", sp.cos: "Derivada del coseno", sp.tan: "Derivada de la tangente",
               sp.exp: "Derivada de la exponencial", sp.log: "Derivada del logaritmo natural",
               sp.asin: "Derivada del arcoseno", sp.acos: "Derivada del arcocoseno",
               sp.atan: "Derivada del arcotangente", sp.sinh: "Derivada del seno hiperbólico",
               sp.cosh: "Derivada del coseno hiperbólico", sp.tanh: "Derivada de la tangente hiperbólica"}


def derivative_steps(expr, x, max_steps=40):
    steps = []

    def add(title, latex):
        if len(steps) < max_steps:
            steps.append(step(title, latex))

    def d(e):
        if not e.has(x):
            add("Derivada de una constante", rf"\frac{{d}}{{d{L(x)}}}\left({L(e)}\right)=0")
            return sp.Integer(0)
        if e == x:
            add("Derivada de la variable", rf"\frac{{d}}{{d{L(x)}}}\left({L(x)}\right)=1")
            return sp.Integer(1)
        if e.is_Add:
            add("Regla de la suma", rf"\left({L(e)}\right)'=" + "+".join(rf"\left({L(t)}\right)'" for t in e.args))
            return sp.Add(*[d(t) for t in e.args])
        if e.is_Mul:
            c, rest = e.as_independent(x)
            if c != 1:
                add("Constante por función", rf"\left({L(c)}\cdot {L(rest)}\right)'={L(c)}\cdot\left({L(rest)}\right)'")
                return c * d(rest)
            f, g = e.args[0], sp.Mul(*e.args[1:])
            # cociente: f * g**-n
            if g.is_Pow and g.exp.is_negative or (f.is_Pow and f.exp.is_negative):
                num = sp.numer(e); den = sp.denom(e)
                if den.has(x):
                    add("Regla del cociente", rf"\left(\frac{{u}}{{v}}\right)'=\frac{{u'v-uv'}}{{v^2}},\quad u={L(num)},\ v={L(den)}")
                    du, dv = d(num), d(den)
                    return (du * den - num * dv) / den**2
            add("Regla del producto", rf"(uv)'=u'v+uv',\quad u={L(f)},\ v={L(g)}")
            df, dg = d(f), d(g)
            return df * g + f * dg
        if e.is_Pow:
            b, n = e.args
            if not n.has(x):
                if b == x:
                    add("Regla de la potencia", rf"\left({L(x)}^{{{L(n)}}}\right)'={L(n)}\,{L(x)}^{{{L(n-1)}}}")
                    return n * x ** (n - 1)
                add("Regla de la potencia con regla de la cadena", rf"\left(u^{{{L(n)}}}\right)'={L(n)}\,u^{{{L(n-1)}}}\,u',\quad u={L(b)}")
                return n * b ** (n - 1) * d(b)
            if not b.has(x):
                add("Derivada de una exponencial", rf"\left({L(b)}^{{u}}\right)'={L(b)}^{{u}}\ln({L(b)})\,u',\quad u={L(n)}")
                return e * sp.log(b) * d(n)
            add("Derivación logarítmica", rf"y={L(e)}\Rightarrow \ln y={L(n)}\ln({L(b)})")
            return sp.diff(e, x)
        if len(e.args) == 1 and e.func in _FUNC_RULES:
            u = e.args[0]
            if u == x:
                add(_FUNC_RULES[e.func], rf"\left({L(e)}\right)'={L(sp.diff(e, x))}")
                return sp.diff(e, x)
            add(_FUNC_RULES[e.func] + " con regla de la cadena", rf"\left({L(e.func(sp.Symbol('u')))}\right)'=({L(sp.diff(e.func(sp.Symbol('u')), sp.Symbol('u')))})\,u',\quad u={L(u)}")
            return sp.diff(e.func(sp.Symbol("u")), sp.Symbol("u")).subs(sp.Symbol("u"), u) * d(u)
        add("Se deriva directamente", rf"\frac{{d}}{{d{L(x)}}}\left({L(e)}\right)={L(sp.diff(e, x))}")
        return sp.diff(e, x)

    res = d(expr)
    final = sp.simplify(res)
    add("Simplificar el resultado", rf"{L(final)}")
    return steps, final


# ------------------------------------------------------------------ integrales
_RULE_ES = {"ConstantRule": "Integral de una constante", "ConstantTimesRule": "Sacar la constante",
            "PowerRule": "Regla de la potencia", "AddRule": "Linealidad (suma)", "URule": "Sustitución",
            "PartsRule": "Integración por partes", "CyclicPartsRule": "Integración por partes cíclica",
            "TrigRule": "Integral trigonométrica básica", "ExpRule": "Integral exponencial",
            "ReciprocalRule": "Integral de 1/x", "ArctanRule": "Integral tipo arcotangente",
            "RewriteRule": "Reescribir el integrando", "AlternativeRule": "Método alternativo",
            "DontKnowRule": "Sin método elemental", "PiecewiseRule": "Por tramos", "HyperbolicRule": "Integral hiperbólica",
            "SqrtQuadraticRule": "Raíz de cuadrática", "DivRule": "División", "NestedPowRule": "Potencia anidada",
            "AtomRule": "Integral de la variable"}


def integral_steps(expr, x, max_steps=30):
    steps = []

    def walk(rule, depth=0):
        if len(steps) >= max_steps or depth > 8:
            return
        name = type(rule).__name__
        title = _RULE_ES.get(name, name)
        integrand = getattr(rule, "integrand", None)
        lat = rf"\int {L(integrand)}\,d{L(x)}" if integrand is not None else ""
        extra = ""
        if name == "URule":
            extra = rf",\quad u={L(rule.u_func)}"
        elif name == "PartsRule":
            extra = rf",\quad u={L(rule.u)},\ dv={L(rule.dv)}"
        steps.append(step(title, lat + extra))
        for attr in ("substep", "second_step", "v_step"):
            s = getattr(rule, attr, None)
            if s is not None and attr != "v_step":
                walk(s, depth + 1)
        for s in getattr(rule, "substeps", []) or []:
            walk(s, depth + 1)

    try:
        rule = mi.integral_steps(expr, x)
        if type(rule).__name__ == "AlternativeRule":
            rule = rule.alternatives[0]
        walk(rule)
    except Exception:
        steps = []
    return steps


# ------------------------------------------------------------------ ecuaciones
def solve_steps(f, x):
    """f(x)=0. Devuelve pasos explicativos (lineal, cuadrática, factorizable, general)."""
    steps = [step("Llevar todo a un lado", rf"{L(sp.expand(f))}=0")]
    try:
        poly = sp.Poly(sp.expand(f), x) if f.is_polynomial(x) else None
    except Exception:
        poly = None
    if poly is None:
        steps.append(step("Ecuación no polinómica", text="Se resuelve de forma simbólica; si no hay solución exacta se buscan raíces numéricas."))
        return steps
    deg = poly.degree()
    cs = poly.all_coeffs()
    if deg == 1:
        a, b = cs
        steps.append(step("Despejar la incógnita", rf"{L(a)}\,{L(x)}={L(-b)}"))
        steps.append(step("Dividir entre el coeficiente", rf"{L(x)}=\frac{{{L(-b)}}}{{{L(a)}}}={L(sp.nsimplify(-b / a))}"))
    elif deg == 2:
        a, b, c = cs
        disc = sp.simplify(b**2 - 4 * a * c)
        steps.append(step("Identificar coeficientes", rf"a={L(a)},\ b={L(b)},\ c={L(c)}"))
        steps.append(step("Discriminante", rf"\Delta=b^2-4ac={L(disc)}"))
        kind = "dos soluciones reales distintas" if disc.is_positive else ("una solución real doble" if disc == 0 else ("dos soluciones complejas conjugadas" if disc.is_negative else ""))
        if kind:
            steps.append(step("Interpretación", text=f"Δ {'> 0' if disc.is_positive else ('= 0' if disc == 0 else '< 0')}: {kind}."))
        steps.append(step("Fórmula general", rf"{L(x)}=\frac{{-b\pm\sqrt{{\Delta}}}}{{2a}}=\frac{{{L(-b)}\pm\sqrt{{{L(disc)}}}}}{{{L(2*a)}}}"))
    else:
        fac = sp.factor(f)
        if fac != sp.expand(f) and fac != f:
            steps.append(step("Factorizar", rf"{L(fac)}=0"))
            steps.append(step("Cada factor igual a cero", text="Un producto es cero cuando alguno de sus factores es cero."))
        else:
            steps.append(step("Polinomio de grado " + str(deg), text="No se factoriza de forma sencilla: se obtienen las raíces con métodos simbólicos/numéricos."))
    return steps


# ------------------------------------------------------------------ Gauss-Jordan
def gauss_jordan(M, ncoef=None, max_steps=40):
    M = sp.Matrix(M)
    rows, cols = M.shape
    steps, r = [], 0
    simp = lambda v: sp.nsimplify(sp.simplify(v)) if v.is_number else sp.simplify(v)
    steps.append(step("Matriz inicial", rf"{L(M)}"))
    for c in range(ncoef or cols):
        if r >= rows:
            break
        piv = next((i for i in range(r, rows) if M[i, c] != 0), None)
        if piv is None:
            continue
        if piv != r:
            M.row_swap(piv, r)
            steps.append(step(f"Intercambiar F{r+1} ↔ F{piv+1}", rf"{L(M)}"))
        pv = M[r, c]
        if pv != 1:
            M[r, :] = (M[r, :] / pv).applyfunc(simp)
            steps.append(step(f"F{r+1} ← F{r+1} / ({pv})", rf"{L(M)}"))
        for i in range(rows):
            if i != r and M[i, c] != 0:
                fct = M[i, c]
                M[i, :] = (M[i, :] - fct * M[r, :]).applyfunc(simp)
                steps.append(step(f"F{i+1} ← F{i+1} − ({fct})·F{r+1}", rf"{L(M)}"))
        r += 1
    return M, steps[:max_steps]
