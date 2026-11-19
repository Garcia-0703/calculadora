"""Analizador SEGURO de expresiones matemáticas (nunca se hace eval de texto arbitrario).

Capas de defensa: 1) normalización de símbolos, 2) validación de tokens (lista blanca de nombres y
operadores; sin atributos, cadenas ni guiones bajos), 3) estimación de tamaño sobre el árbol SIN evaluar
(evita 9**9**9, 100000! ...), 4) análisis con __builtins__ vacío.
"""
import io
import keyword
import math
import re
import tokenize

import sympy as sp
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application, parse_expr,
                                        rationalize, standard_transformations)

from .errors import CalcError

MAX_LEN = 400
MAX_DIGITS_RESULT = 4000          # tamaño máximo estimado (dígitos) de un resultado entero/racional
TRANSFORMS = standard_transformations + (rationalize, implicit_multiplication_application, convert_xor)
ALLOWED_OPS = {"+", "-", "*", "/", "**", "^", "(", ")", ",", "//", "!"}
_HELPERS = ["Integer", "Float", "Rational", "Symbol", "Function", "Add", "Mul", "Pow", "factorial",
            "factorial2", "Tuple", "Number"]


def _log10(x, b=10):
    return sp.log(x, b)


def _limited(f, name):
    """Envuelve funciones que explotan con argumentos grandes (factorial, gamma, binomial)."""
    def g(*args):
        for a in args:
            a = sp.sympify(a)
            if a.is_Number and abs(a) > 3000:
                raise CalcError(f"{name}: el argumento es demasiado grande (máx. 3000).")
        return f(*args)
    return g


def namespace(angle="rad"):
    """Funciones y constantes permitidas. En modo 'deg' las trigonométricas trabajan en grados."""
    deg = angle == "deg"
    k = sp.pi / 180

    def tr(f):
        return (lambda x: f(x * k)) if deg else f

    def itr(f):
        return (lambda x: f(x) / k) if deg else f

    ns = dict(
        sin=tr(sp.sin), cos=tr(sp.cos), tan=tr(sp.tan), cot=tr(sp.cot), sec=tr(sp.sec), csc=tr(sp.csc),
        asin=itr(sp.asin), acos=itr(sp.acos), atan=itr(sp.atan), acot=itr(sp.acot),
        atan2=(lambda y, x: sp.atan2(y, x) / k) if deg else sp.atan2,
        sinh=sp.sinh, cosh=sp.cosh, tanh=sp.tanh, asinh=sp.asinh, acosh=sp.acosh, atanh=sp.atanh,
        sqrt=sp.sqrt, cbrt=sp.cbrt, root=sp.root, exp=sp.exp, ln=sp.log, log=_log10, log10=_log10,
        log2=lambda x: sp.log(x, 2), abs=sp.Abs, Abs=sp.Abs, sign=sp.sign, floor=sp.floor,
        ceil=sp.ceiling, ceiling=sp.ceiling, factorial=_limited(sp.factorial, 'factorial'),
        gamma=_limited(sp.gamma, 'gamma'), binomial=_limited(sp.binomial, 'binomial'),
        gcd=sp.gcd, lcm=sp.lcm, re=sp.re, im=sp.im, arg=(lambda z: sp.arg(z) / k) if deg else sp.arg,
        conj=sp.conjugate, conjugate=sp.conjugate, Min=sp.Min, Max=sp.Max, min=sp.Min, max=sp.Max,
        mod=sp.Mod, pi=sp.pi, e=sp.E, E=sp.E, i=sp.I, I=sp.I, oo=sp.oo, inf=sp.oo, phi=sp.GoldenRatio)
    return ns


def _globals(ns):
    g = {"__builtins__": {}}
    g.update({h: getattr(sp, h) for h in _HELPERS})
    g.update(ns)
    return g


def normalize(s, angle="rad"):
    s = (s or "").strip()
    for a, b in {"×": "*", "·": "*", "÷": "/", "−": "-", "–": "-", "π": "pi", "²": "**2", "³": "**3",
                 "∞": "oo", "≤": "<=", "≥": ">=", "≠": "!=", "√(": "sqrt("}.items():
        s = s.replace(a, b)
    s = re.sub(r"√\s*(\d+(?:\.\d+)?|[A-Za-z]\w*)", r"sqrt(\1)", s)
    s = s.replace("°", "" if angle == "deg" else "*(pi/180)")
    return s


_VAR = re.compile(r"^(?:[A-Za-z][A-Za-z0-9]{0,2}|[A-Za-z]_[A-Za-z0-9]{1,2})$")


def _validate(s, ns):
    if not s:
        raise CalcError("Escribe una expresión.")
    if len(s) > MAX_LEN:
        raise CalcError(f"La expresión es demasiado larga (máximo {MAX_LEN} caracteres).")
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(s).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        raise CalcError("Expresión mal formada: revisa los paréntesis y los operadores.")
    for t in toks:
        if t.type == tokenize.NAME:
            n = t.string
            if keyword.iskeyword(n):
                raise CalcError(f"Palabra no permitida: «{n}».")
            if n not in ns and not (_VAR.match(n) or (len(n) == 1 and n.isalpha())):
                raise CalcError(f"Nombre no reconocido: «{n}». Usa variables de 1–3 letras o funciones conocidas (sin, ln, sqrt…).")
        elif t.type == tokenize.OP:
            if t.string not in ALLOWED_OPS:
                raise CalcError(f"Símbolo no permitido: «{t.string}».")
        elif t.type == tokenize.ERRORTOKEN:
            if t.string.strip() not in ("!", ""):
                raise CalcError(f"Símbolo no permitido: «{t.string}».")
        elif t.type == tokenize.STRING:
            raise CalcError("No se permiten cadenas de texto.")
        elif t.type == tokenize.NUMBER and len(t.string) > 300:
            raise CalcError("Número demasiado largo.")


def _est(node):
    """Magnitud estimada (float) de un subárbol numérico sin evaluar; None si es simbólico."""
    if node.is_Number:
        try:
            return float(node)
        except OverflowError:
            return math.inf
    if node.is_Atom:
        return None
    args = [_est(a) for a in node.args]
    if any(a is None for a in args):
        return None
    try:
        if node.is_Pow:
            b, e = args
            if b == 0 or b == 1 or e == 0:
                return 1.0
            digits = abs(e) * math.log10(abs(b)) if abs(b) > 0 else 0
            if digits > MAX_DIGITS_RESULT:
                raise CalcError("El resultado sería enorme (potencia demasiado grande).")
            return float(b) ** float(e)
        if node.is_Mul:
            r = 1.0
            for a in args:
                r *= a
            return r
        if node.is_Add:
            return sum(args)
        if node.func.__name__ in ("factorial", "gamma", "binomial", "factorial2"):
            if any(abs(a) > 3000 for a in args if a == a):
                raise CalcError("Factorial/gamma/binomial: el argumento es demasiado grande (máx. 3000).")
            return 1.0
    except OverflowError:
        raise CalcError("El resultado sería enorme (desbordamiento).")
    return 1.0


def _guard(s, ns):
    g = _globals(ns)
    try:
        tree = parse_expr(s, global_dict=g, local_dict={}, transformations=TRANSFORMS, evaluate=False)
    except CalcError:
        raise
    except Exception:
        return  # el análisis real informará del error
    try:
        _est(tree)
    except RecursionError:
        raise CalcError("Expresión demasiado anidada.")


def parse(text, angle="rad", extra=None):
    """Texto -> expresión de sympy (evaluada). Lanza CalcError ante cualquier entrada no válida."""
    s = normalize(text, angle)
    ns = namespace(angle)
    _validate(s, ns)
    _guard(s, ns)
    loc = dict(extra or {})
    try:
        return parse_expr(s, global_dict=_globals(ns), local_dict=loc, transformations=TRANSFORMS, evaluate=True)
    except CalcError:
        raise
    except (SyntaxError, TypeError, ValueError, AttributeError, NameError, ZeroDivisionError, sp.SympifyError,
            sp.polys.polyerrors.PolynomialError, tokenize.TokenError) as e:
        raise CalcError(f"No pude interpretar la expresión: {type(e).__name__}. Revisa la sintaxis.")
    except RecursionError:
        raise CalcError("Expresión demasiado anidada.")


def parse_unevaluated(text, angle="rad"):
    s = normalize(text, angle)
    ns = namespace(angle)
    _validate(s, ns)
    _guard(s, ns)
    try:
        return parse_expr(s, global_dict=_globals(ns), local_dict={}, transformations=TRANSFORMS, evaluate=False)
    except Exception:
        return None


_REL = re.compile(r"(<=|>=|!=|==|<|>|=)")


def split_relation(text):
    """'a <= b' -> ('a', '<=', 'b'); sin relación -> None."""
    parts = _REL.split(normalize(text))
    if len(parts) == 1:
        return None
    if len(parts) != 3:
        raise CalcError("Solo se admite una relación (=, <, >, ≤, ≥) por línea.")
    return parts[0].strip(), parts[1], parts[2].strip()


def parse_relation(text, angle="rad"):
    """Devuelve (expr_lhs, op, expr_rhs) con op en {'=','<','>','<=','>=','!='} o (expr, None, None)."""
    rel = split_relation(text)
    if rel is None:
        return parse(text, angle), None, None
    l, op, r = rel
    if not l or not r:
        raise CalcError("Falta un lado de la ecuación/inecuación.")
    return parse(l, angle), ("=" if op == "==" else op), parse(r, angle)
