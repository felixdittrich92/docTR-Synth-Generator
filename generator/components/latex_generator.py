# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

import io
import random

__all__ = ["generate_latex_formulas", "render_latex_image", "is_valid_latex"]

# Building blocks, all within the mathtext-supported subset of LaTeX.
_GREEK = [
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
    r"\theta",
    r"\lambda",
    r"\mu",
    r"\pi",
    r"\rho",
    r"\sigma",
    r"\phi",
    r"\omega",
    r"\psi",
    r"\tau",
]
_VARS = list("abcdefghijklmnpqrstuvxyz") + list("ABCDEFGHJKLMNPQRSTUVWXYZ")
_FUNCS = [r"\sin", r"\cos", r"\tan", r"\log", r"\ln", r"\exp"]
_REL = ["=", r"\leq", r"\geq", r"\neq", r"\approx", "<", ">", r"\equiv"]
_BINOP = ["+", "-", r"\times", r"\cdot", r"\div", r"\pm"]
_CONST = [r"\infty", r"\pi", r"0", r"1", r"2", r"n", r"k", r"x"]


def _num(rng: random.Random) -> str:
    return str(rng.randint(0, 99))


def _atom(rng: random.Random) -> str:
    """A leaf: a variable, Greek letter, or number."""
    return rng.choice(_VARS + _GREEK + [_num(rng), _num(rng)])


def _term(rng: random.Random, depth: int) -> str:
    """A sub-expression, recursively, capped by ``depth``."""
    if depth <= 0:
        return _atom(rng)
    kind = rng.choice(["atom", "atom", "pow", "sub", "frac", "sqrt", "func", "paren", "sum", "int"])
    if kind == "atom":
        return _atom(rng)
    if kind == "pow":
        return f"{_atom(rng)}^{{{_term(rng, depth - 1)}}}"
    if kind == "sub":
        return f"{_atom(rng)}_{{{_term(rng, depth - 1)}}}"
    if kind == "frac":
        return f"\\frac{{{_term(rng, depth - 1)}}}{{{_term(rng, depth - 1)}}}"
    if kind == "sqrt":
        return f"\\sqrt{{{_term(rng, depth - 1)}}}"
    if kind == "func":
        return f"{rng.choice(_FUNCS)}({_term(rng, depth - 1)})"
    if kind == "paren":
        return f"({_expr(rng, depth - 1)})"
    if kind == "sum":
        return f"\\sum_{{i=1}}^{{{rng.choice(['n', 'k', 'N'])}}} {_term(rng, depth - 1)}"
    if kind == "int":
        upper = rng.choice(["1", "n", r"\infty"])
        return f"\\int_{{0}}^{{{upper}}} {_term(rng, depth - 1)}\\,dx"
    return _atom(rng)


def _expr(rng: random.Random, depth: int) -> str:
    """A chain of terms joined by binary operators."""
    parts = [_term(rng, depth)]
    for _ in range(rng.randint(0, 2)):
        parts.append(rng.choice(_BINOP))
        parts.append(_term(rng, depth))
    return " ".join(parts)


def _formula(rng: random.Random, max_depth: int) -> str:
    """A full formula: usually a relation ``lhs <rel> rhs``."""
    lhs = _expr(rng, max_depth)
    if rng.random() < 0.7:
        return f"{lhs} {rng.choice(_REL)} {_expr(rng, max_depth)}"
    return lhs


# A single reusable parser; mathtext raises on anything it cannot render.
_PARSER = None


def is_valid_latex(formula: str) -> bool:
    """True if mathtext can parse (and therefore render) ``$formula$``."""
    global _PARSER
    if _PARSER is None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.mathtext import MathTextParser

        _PARSER = MathTextParser("agg")
    try:
        _PARSER.parse(f"${formula}$")
        return True
    except Exception:
        return False


def generate_latex_formulas(
    n: int, *, seed: int | None = None, max_depth: int = 3, min_len: int = 3, max_len: int = 60
) -> list[str]:
    """Return ``n`` distinct, valid LaTeX math source strings (no surrounding ``$``)."""
    rng = random.Random(seed)
    out: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 40:
        attempts += 1
        f = _formula(rng, max_depth)
        if not (min_len <= len(f) <= max_len) or f in seen:
            continue
        if is_valid_latex(f):
            seen.add(f)
            out.append(f)
    return out


def render_latex_image(formula: str, *, fontsize: int = 24, dpi: int = 200, color=(0, 0, 0)):
    """Render ``formula`` to a tightly-cropped RGBA ``PIL.Image`` (transparent bg)."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from PIL import Image

    fig = plt.figure()
    rgb = tuple(c / 255 for c in color)
    fig.text(0.0, 0.0, f"${formula}$", fontsize=fontsize, color=rgb)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")
