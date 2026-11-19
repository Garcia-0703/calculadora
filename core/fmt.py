"""Formato de salida: LaTeX y aproximaciones decimales con la precisión pedida."""
import re
import sympy as sp


def L(e):
    try:
        return sp.latex(e, inv_trig_style="full")
    except Exception:
        return str(e)


def _strip(s):
    if "e" in s or "E" in s:
        m, _, ex = re.split(r"(e|E)", s, maxsplit=1)[0], None, re.split(r"(e|E)", s, maxsplit=1)[2]
        m = _strip(m)
        return f"{m}e{int(ex):+d}".replace("e+", "e+")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def num_text(x, digits=15):
    """Texto decimal de un número real de sympy con 'digits' cifras significativas."""
    v = sp.N(x, digits)
    if v == 0:
        return "0"
    s = str(v)
    return _strip(s)


def approx_text(e, digits=15):
    """Aproximación decimal (real o compleja) o None si hay símbolos libres."""
    if e.free_symbols:
        return None
    if e in (sp.oo, -sp.oo, sp.zoo, sp.nan):
        return str(e).replace("zoo", "∞ complejo").replace("oo", "∞")
    re_, im_ = e.as_real_imag() if not e.is_real else (e, sp.Integer(0))
    re_, im_ = sp.N(re_, digits + 5), sp.N(im_, digits + 5)
    if im_ == 0 or abs(im_) < sp.Float(10) ** (-digits - 2):
        return num_text(re_, digits)
    a, b = num_text(re_, digits), num_text(abs(im_), digits)
    sign = "−" if im_ < 0 else "+"
    if a == "0":
        return f"{'-' if im_ < 0 else ''}{b}i"
    return f"{a} {sign} {b}i"


def block(label, latex=None, text=None):
    d = {"label": label}
    if latex is not None:
        d["latex"] = latex
    if text is not None:
        d["text"] = text
    return d


def step(title, latex=None, text=None):
    d = {"title": title}
    if latex is not None:
        d["latex"] = latex
    if text is not None:
        d["text"] = text
    return d
