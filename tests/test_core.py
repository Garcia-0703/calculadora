"""Pruebas del núcleo: python -m unittest discover -s tests"""
import sys, os, json, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import calc, algebra, solver, calculus, matrices, stats, graph, units, service
from core.errors import CalcError
from core.parser import parse


def txt(d, label_start):
    return next(b for b in d["blocks"] if b["label"].startswith(label_start))


class TestParser(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(str(parse("2+3*4")), "14")
        self.assertEqual(str(parse("5!")), "120")
        self.assertEqual(str(parse("0.1+0.2")), "3/10")          # decimales exactos
        self.assertEqual(str(parse("sin(30)", "deg")), "1/2")
        self.assertEqual(str(parse("e^(i*pi)")), "-1")

    def test_security(self):
        for bad in ["__import__('os').system('ls')", "().__class__", "x.y", "lambda: 1", "open('f')", "exec('1')",
                    "9**9**9", "2**100000", "factorial(100000)", "99999!", "[1,2]", "'a'", "x;y", "import os", "a" * 500, "sin("]:
            with self.assertRaises(CalcError, msg=bad):
                parse(bad)


class TestCalc(unittest.TestCase):
    def test_exact_and_decimal(self):
        d = calc.evaluate("sqrt(2)", digits=30)
        self.assertEqual(txt(d, "Aproximación")["text"], "1.41421356237309504880168872421")
        self.assertEqual(calc.evaluate("1/3")["blocks"][0]["latex"], r"\frac{1}{3}")

    def test_complex_and_errors(self):
        d = calc.evaluate("(3+4i)*(1-2i)")
        self.assertEqual(txt(d, "Resultado")["latex"], "11 - 2 i")
        self.assertIn("Módulo", [b["label"] for b in d["blocks"]][2])
        with self.assertRaises(CalcError): calc.evaluate("1/0")

    def test_steps(self):
        d = calc.evaluate("(3+4)*2^3-5")
        self.assertEqual(d["steps"][-1]["latex"], "51")


class TestSolver(unittest.TestCase):
    def test_quadratic_steps(self):
        d = solver.run("x^2-5x+6=0")
        self.assertEqual([b["latex"] for b in d["blocks"][1:]], ["x_{1} = 2", "x_{2} = 3"])
        self.assertTrue(any("Discriminante" in s["title"] for s in d["steps"]))

    def test_system_and_inequality(self):
        d = solver.run("x+y=3\n2x-y=0")
        self.assertEqual([b["latex"] for b in d["blocks"][1:]], ["x = 1", "y = 2"])
        d = solver.run("x^2-4>=0")
        self.assertIn(r"\left(-\infty, -2\right] \cup \left[2, \infty\right)", txt(d, "Solución")["latex"])

    def test_numeric_fallback(self):
        d = solver.run("cos(x)=x")
        self.assertIn("0.739085", txt(d, "Raíces")["text"])


class TestCalculus(unittest.TestCase):
    def test_derivative(self):
        d = calculus.derivative("x^2*sin(x)")
        self.assertEqual(d["steps"][0]["title"], "Regla del producto")
        self.assertEqual(txt(calculus.derivative("(x^2+1)/(x-3)", at="5"), "Evaluada")["latex"], r"- \frac{3}{2}")

    def test_integrals(self):
        self.assertEqual(txt(calculus.integral("sin(x)", a="0", b="pi"), "Valor")["latex"], "2")
        self.assertEqual(txt(calculus.integral("e^(-x^2)", a="-oo", b="oo"), "Valor")["latex"], r"\sqrt{\pi}")
        with self.assertRaises(CalcError): calculus.integral("x^x^x")

    def test_limit(self):
        self.assertIn("= 1", txt(calculus.limit("sin(x)/x"), "Límite")["latex"])


class TestMatrices(unittest.TestCase):
    def test_ops(self):
        self.assertIn("-2", txt(matrices.run("det", "1 2\n3 4"), "Determinante")["latex"])
        self.assertIn("37", txt(matrices.run("pow", "1 2\n3 4", k="3"), "A^3")["latex"])
        with self.assertRaises(CalcError): matrices.run("inv", "1 2\n2 4")
        with self.assertRaises(CalcError): matrices.run("det", "1 2 3\n4 5 6")


class TestStatsGraphUnits(unittest.TestCase):
    def test_stats(self):
        d = {s["name"]: s["value"] for s in stats.descriptive("2 4 4 4 5 5 7 9")["stats"]}
        self.assertEqual(d["Media"], 5.0)
        self.assertEqual(d["Desviación estándar poblacional σ"], 2.0)

    def test_regression(self):
        r = stats.regression("1 2 3 4 5", "2 4 6 8 10", "linear")
        self.assertAlmostEqual(r["r2"], 1.0, places=9)
        with self.assertRaises(CalcError): stats.regression("1 2 3", "1 2", "linear")

    def test_graph_and_units(self):
        g = graph.graph2d(["1/x"], -5, 5)
        json.dumps(g, allow_nan=False)
        with self.assertRaises(CalcError): graph.graph2d(["x+z"])
        self.assertEqual(units.convert("Longitud", 1, "mi", "km")["text"], "1.609344")
        self.assertEqual(units.convert("Temperatura", 100, "°C", "°F")["text"], "212")
        with self.assertRaises(CalcError): units.convert("Temperatura", -500, "°C", "K")

    def test_service_wraps_unexpected(self):
        with self.assertRaises(CalcError): service.calc_eval(dict(expr="((("))


if __name__ == "__main__":
    unittest.main()
