"""Matrices: operaciones y pasos de Gauss-Jordan."""
import re
import sympy as sp
from .errors import CalcError
from .fmt import L, approx_text, block, step
from .parser import parse
from . import steps as st

MAXN = 8


def parse_matrix(text, name="A"):
    rows = [r for r in re.split(r"[\n;]+", text or "") if r.strip()]
    if not rows:
        raise CalcError(f"Escribe la matriz {name} (filas separadas por salto de línea o «;», entradas por espacios o comas).")
    data = []
    for r in rows:
        ents = [t for t in re.split(r"[,\s]+", r.strip()) if t]
        data.append([parse(t) for t in ents])
    w = {len(r) for r in data}
    if len(w) != 1:
        raise CalcError(f"La matriz {name} tiene filas de distinto tamaño.")
    if len(data) > MAXN or len(data[0]) > MAXN:
        raise CalcError(f"Tamaño máximo {MAXN}×{MAXN}.")
    return sp.Matrix(data)


def _square(M, name="A"):
    if M.rows != M.cols:
        raise CalcError(f"La matriz {name} debe ser cuadrada para esta operación.")


def _det_steps(M):
    n = M.rows
    if n == 1:
        return [step("Determinante 1×1", L(M[0, 0]))]
    if n == 2:
        a, b, c, d = M
        return [step("Fórmula 2×2: ad − bc", rf"{L(a)}\cdot{L(d)}-{L(b)}\cdot{L(c)}={L(sp.simplify(a*d-b*c))}")]
    terms = []
    for j in range(n):
        minor = M.minor_submatrix(0, j)
        sign = "+" if j % 2 == 0 else "-"
        terms.append((sign, M[0, j], minor))
    s = [step("Desarrollo por cofactores (primera fila)", " ".join(f"{sg}\\,{L(a)}\\,\\det{L(mn)}" for sg, a, mn in terms))]
    s.append(step("Valores de los menores", "+".join(f"({'-1' if j%2 else '1'})\\cdot{L(M[0,j])}\\cdot({L(M.minor(0,j))})" for j in range(n))))
    return s


def run(op, A, B=None, k=None, digits=15):
    M = parse_matrix(A, "A")
    N = parse_matrix(B, "B") if B and B.strip() else None
    blocks, steps = [block("Matriz A", latex=L(M))], []
    if N is not None:
        blocks.append(block("Matriz B", latex=L(N)))

    def need_b():
        if N is None:
            raise CalcError("Esta operación necesita la matriz B.")

    if op == "det":
        _square(M)
        d = sp.simplify(M.det())
        steps = _det_steps(M)
        blocks.append(block("Determinante", latex=rf"\det(A)={L(d)}", text=approx_text(d, digits) if not d.is_Integer else None))
    elif op == "inv":
        _square(M)
        if sp.simplify(M.det()) == 0:
            raise CalcError("La matriz es singular (determinante 0): no tiene inversa.")
        aug = M.row_join(sp.eye(M.rows))
        R, steps = st.gauss_jordan(aug, ncoef=M.cols)
        inv = sp.simplify(M.inv())
        blocks.append(block("Inversa A⁻¹", latex=rf"A^{{-1}}={L(inv)}"))
        blocks.append(block("Determinante", latex=L(sp.simplify(M.det()))))
    elif op == "transpose":
        blocks.append(block("Transpuesta", latex=rf"A^{{T}}={L(M.T)}"))
    elif op == "trace":
        _square(M)
        blocks.append(block("Traza", latex=rf"\operatorname{{tr}}(A)={L(sp.simplify(M.trace()))}"))
    elif op == "rank":
        blocks.append(block("Rango", latex=rf"\operatorname{{rango}}(A)={M.rank()}"))
        _, steps = st.gauss_jordan(M)
    elif op == "rref":
        R, steps = st.gauss_jordan(M)
        blocks.append(block("Forma escalonada reducida", latex=L(R)))
    elif op == "eigen":
        _square(M)
        lam = sp.symbols("lambda")
        cp = sp.factor(M.charpoly(lam).as_expr())
        steps.append(step("Polinomio característico det(A − λI)", rf"{L(cp)}=0"))
        try:
            ev = M.eigenvects()
        except Exception:
            raise CalcError("No pude calcular los autovalores de esta matriz.")
        for val, mult, vecs in ev:
            blocks.append(block(f"Autovalor λ = {sp.latex(val)} (mult. {mult})", latex=rf"\lambda={L(sp.simplify(val))},\quad " + r",\ ".join(rf"v={L(sp.simplify(v))}" for v in vecs), text=approx_text(val, digits) if val.is_number and not val.is_Rational else None))
    elif op == "charpoly":
        _square(M)
        lam = sp.symbols("lambda")
        blocks.append(block("Polinomio característico", latex=L(sp.expand(M.charpoly(lam).as_expr()))))
    elif op == "nullspace":
        ns = M.nullspace()
        blocks.append(block("Núcleo (base)", latex=r",\ ".join(L(v) for v in ns) if ns else r"\{0\}"))
        blocks.append(block("Nulidad", text=str(len(ns))))
    elif op in ("add", "sub", "mul"):
        need_b()
        try:
            r = {"add": M + N, "sub": M - N, "mul": M * N}[op]
        except sp.matrices.exceptions.ShapeError:
            raise CalcError("Las dimensiones no son compatibles para esta operación.")
        blocks.append(block({"add": "A + B", "sub": "A − B", "mul": "A · B"}[op], latex=L(sp.simplify(r))))
    elif op == "pow":
        _square(M)
        n = int(parse(str(k if k is not None else 2)))
        if abs(n) > 50:
            raise CalcError("El exponente debe estar entre −50 y 50.")
        blocks.append(block(f"A^{n}", latex=L(sp.simplify(M**n))))
    elif op == "scalar":
        c = parse(str(k if k is not None else 1))
        blocks.append(block(f"{L(c)}·A", latex=L(sp.simplify(c * M))))
    elif op == "solve":
        need_b()
        if N.cols != 1 and M.rows != N.rows:
            raise CalcError("B debe tener el mismo número de filas que A.")
        if M.rows != N.rows:
            raise CalcError("B debe tener el mismo número de filas que A.")
        aug = M.row_join(N)
        R, steps = st.gauss_jordan(aug, ncoef=M.cols)
        syms = sp.symbols(f"x1:{M.cols + 1}")
        sol = sp.linsolve((M, N[:, 0]), *syms)
        if sol == sp.EmptySet:
            blocks.append(block("Resultado", text="Sistema incompatible (sin solución)"))
        else:
            tup = list(sol)[0]
            blocks.append(block("Solución X", latex=r"X=" + L(sp.Matrix(tup))))
    else:
        raise CalcError("Operación de matrices no soportada.")
    return dict(blocks=blocks, steps=steps)
