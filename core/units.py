"""Conversión de unidades y constantes físicas/matemáticas (valores CODATA 2018)."""
from .errors import CalcError

# categoría -> (unidad base, {unidad: factor a base})   (la temperatura se trata aparte)
CATS = {
    "Longitud": {"m": 1, "mm": 1e-3, "cm": 1e-2, "km": 1e3, "µm": 1e-6, "nm": 1e-9, "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344, "nmi": 1852, "ly": 9.4607304725808e15, "au": 1.495978707e11},
    "Masa": {"kg": 1, "g": 1e-3, "mg": 1e-6, "t": 1e3, "lb": 0.45359237, "oz": 0.028349523125, "st": 6.35029318, "u": 1.66053906660e-27},
    "Tiempo": {"s": 1, "ms": 1e-3, "µs": 1e-6, "min": 60, "h": 3600, "día": 86400, "semana": 604800, "año": 31557600},
    "Velocidad": {"m/s": 1, "km/h": 1 / 3.6, "mph": 0.44704, "kn": 1852 / 3600, "ft/s": 0.3048, "c": 299792458},
    "Área": {"m²": 1, "cm²": 1e-4, "mm²": 1e-6, "km²": 1e6, "ha": 1e4, "in²": 0.00064516, "ft²": 0.09290304, "yd²": 0.83612736, "ac": 4046.8564224, "mi²": 2589988.110336},
    "Volumen": {"m³": 1, "L": 1e-3, "mL": 1e-6, "cm³": 1e-6, "gal (US)": 0.003785411784, "qt (US)": 0.000946352946, "pt (US)": 0.000473176473, "fl oz (US)": 2.95735295625e-5, "taza (US)": 0.0002365882365, "cda": 1.4786764781e-5, "cdta": 4.92892159375e-6, "ft³": 0.028316846592, "in³": 1.6387064e-5},
    "Energía": {"J": 1, "kJ": 1e3, "MJ": 1e6, "cal": 4.184, "kcal": 4184, "Wh": 3600, "kWh": 3.6e6, "eV": 1.602176634e-19, "BTU": 1055.05585262, "erg": 1e-7, "ft·lbf": 1.3558179483314},
    "Potencia": {"W": 1, "kW": 1e3, "MW": 1e6, "hp": 745.69987158227, "CV": 735.49875, "BTU/h": 0.29307107, "cal/s": 4.184},
    "Presión": {"Pa": 1, "kPa": 1e3, "MPa": 1e6, "bar": 1e5, "mbar": 100, "atm": 101325, "psi": 6894.757293168, "mmHg": 133.322387415, "torr": 101325 / 760},
    "Fuerza": {"N": 1, "kN": 1e3, "dyn": 1e-5, "kgf": 9.80665, "lbf": 4.4482216152605},
    "Ángulo": {"rad": 1, "°": 3.141592653589793 / 180, "grad": 3.141592653589793 / 200, "vuelta": 2 * 3.141592653589793, "arcmin": 3.141592653589793 / 10800, "arcsec": 3.141592653589793 / 648000},
    "Frecuencia": {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "rpm": 1 / 60},
    "Datos": {"bit": 1 / 8, "B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4},
}
TEMPS = ["°C", "°F", "K", "°R"]


def catalog():
    d = {k: list(v) for k, v in CATS.items()}
    d["Temperatura"] = TEMPS
    return d


def _to_kelvin(v, u):
    return {"°C": v + 273.15, "K": v, "°F": (v - 32) * 5 / 9 + 273.15, "°R": v * 5 / 9}[u]


def _from_kelvin(k, u):
    return {"°C": k - 273.15, "K": k, "°F": (k - 273.15) * 9 / 5 + 32, "°R": k * 9 / 5}[u]


def convert(category, value, frm, to):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise CalcError("El valor a convertir debe ser un número.")
    if v != v or v in (float("inf"), float("-inf")):
        raise CalcError("El valor debe ser finito.")
    if category == "Temperatura":
        if frm not in TEMPS or to not in TEMPS:
            raise CalcError("Unidad de temperatura no válida.")
        k = _to_kelvin(v, frm)
        if k < -1e-9:
            raise CalcError("Esa temperatura está por debajo del cero absoluto.")
        r = _from_kelvin(k, to)
    else:
        units = CATS.get(category)
        if units is None or frm not in units or to not in units:
            raise CalcError("Unidad no válida para esa categoría.")
        r = v * units[frm] / units[to]
    return dict(result=float(f"{r:.12g}"), text=f"{r:.10g}", from_=frm, to=to)


CONSTANTS = [
    # (nombre, símbolo, valor, unidad, categoría)
    ("Velocidad de la luz", "c", 299792458, "m/s", "Física"),
    ("Constante de Planck", "h", 6.62607015e-34, "J·s", "Física"),
    ("Constante de Planck reducida", "ħ", 1.054571817e-34, "J·s", "Física"),
    ("Constante gravitacional", "G", 6.67430e-11, "m³/(kg·s²)", "Física"),
    ("Aceleración de la gravedad estándar", "g", 9.80665, "m/s²", "Física"),
    ("Carga elemental", "e", 1.602176634e-19, "C", "Física"),
    ("Constante de Boltzmann", "k_B", 1.380649e-23, "J/K", "Física"),
    ("Constante de Avogadro", "N_A", 6.02214076e23, "1/mol", "Química"),
    ("Constante de los gases", "R", 8.314462618, "J/(mol·K)", "Química"),
    ("Constante de Faraday", "F", 96485.33212, "C/mol", "Química"),
    ("Unidad de masa atómica", "u", 1.66053906660e-27, "kg", "Química"),
    ("Permitividad del vacío", "ε₀", 8.8541878128e-12, "F/m", "Electromagnetismo"),
    ("Permeabilidad del vacío", "μ₀", 1.25663706212e-6, "N/A²", "Electromagnetismo"),
    ("Masa del electrón", "m_e", 9.1093837015e-31, "kg", "Partículas"),
    ("Masa del protón", "m_p", 1.67262192369e-27, "kg", "Partículas"),
    ("Masa del neutrón", "m_n", 1.67492749804e-27, "kg", "Partículas"),
    ("Constante de estructura fina", "α", 7.2973525693e-3, "", "Partículas"),
    ("Radio de Bohr", "a₀", 5.29177210903e-11, "m", "Partículas"),
    ("Constante de Rydberg", "R∞", 10973731.568160, "1/m", "Partículas"),
    ("Constante de Stefan–Boltzmann", "σ", 5.670374419e-8, "W/(m²·K⁴)", "Termodinámica"),
    ("Presión atmosférica estándar", "atm", 101325, "Pa", "Termodinámica"),
    ("Cero absoluto", "T₀", -273.15, "°C", "Termodinámica"),
    ("Pi", "π", 3.141592653589793, "", "Matemáticas"),
    ("Número de Euler", "e", 2.718281828459045, "", "Matemáticas"),
    ("Razón áurea", "φ", 1.618033988749895, "", "Matemáticas"),
    ("Raíz cuadrada de 2", "√2", 1.4142135623730951, "", "Matemáticas"),
    ("Constante de Euler–Mascheroni", "γ", 0.5772156649015329, "", "Matemáticas"),
    ("Unidad astronómica", "au", 1.495978707e11, "m", "Astronomía"),
    ("Año luz", "ly", 9.4607304725808e15, "m", "Astronomía"),
    ("Masa del Sol", "M☉", 1.98847e30, "kg", "Astronomía"),
    ("Masa de la Tierra", "M⊕", 5.9722e24, "kg", "Astronomía"),
    ("Radio de la Tierra (medio)", "R⊕", 6.371e6, "m", "Astronomía"),
]


def constants():
    return [dict(name=n, symbol=s, value=v, text=f"{v:.10g}", unit=u, category=c) for n, s, v, u, c in CONSTANTS]
