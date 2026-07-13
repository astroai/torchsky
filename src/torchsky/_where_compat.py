"""Small WHERE parser retained for the published torchfits 0.7 wheel.

Torchfits 0.8 exposes this surface publicly.  The 0.7 wheel still ships the
parser implementation privately, so torchsky keeps this dependency-free
compatibility copy until the public wheel is available.
"""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from typing import Any

import numpy as np

_WHERE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_where_literal(raw: str) -> Any:
    token = raw.strip()
    if not token:
        raise ValueError("where literal cannot be empty")
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        quote = token[0]
        return token[1:-1].replace(f"\\{quote}", quote)
    lower = token.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"none", "null"}:
        return None
    if re.fullmatch(r"[+-]?\d+", token):
        return int(token)
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token):
        return float(token)
    return token


def _tokenize_where_expression(where: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    i = 0
    while i < len(where):
        char = where[i]
        if char.isspace():
            i += 1
            continue
        if char in "(),":
            tokens.append(({"(": "LPAREN", ")": "RPAREN", ",": "COMMA"}[char], char))
            i += 1
            continue
        if i + 1 < len(where) and where[i : i + 2] in {"==", "!=", ">=", "<="}:
            tokens.append(("OP", where[i : i + 2]))
            i += 2
            continue
        if char in "><":
            tokens.append(("OP", char))
            i += 1
            continue
        if char in {"'", '"'}:
            quote = char
            i += 1
            value: list[str] = []
            while i < len(where) and where[i] != quote:
                if where[i] == "\\" and i + 1 < len(where):
                    value.append(where[i + 1])
                    i += 2
                else:
                    value.append(where[i])
                    i += 1
            if i >= len(where):
                raise ValueError("Unterminated quoted literal in where expression")
            i += 1
            tokens.append(("LITERAL", "".join(value)))
            continue
        start = i
        while i < len(where) and not (where[i].isspace() or where[i] in "(),><!="):
            i += 1
        if start == i:
            raise ValueError(f"Unexpected token in where expression near position {i}")
        tokens.append(("WORD", where[start:i]))
    return tokens


def _normalize_where_syntax(where: str) -> str:
    result = where.replace("&&", " AND ").replace("||", " OR ")
    result = re.sub(r"(?<!\w)~(?!\w)", " NOT ", result)
    result = re.sub(r"(?<![!=<>])&(?!&)", " AND ", result)
    return re.sub(r"(?<!\|)\|(?!\|)", " OR ", result)


def _normalize_logical_operators(where: str) -> str:
    parts = re.split(r"('[^']*'|\"[^\"]*\")", where)
    for index in range(0, len(parts), 2):
        part = parts[index]
        for word in ("AND", "OR", "NOT", "IN"):
            part = re.sub(rf"\b{word}\b", word.lower(), part, flags=re.IGNORECASE)
        parts[index] = part
    return "".join(parts)


def _normalize_between(where: str) -> str:
    value = r"('[^']+'|\"[^\"]+\"|\S+)"
    where = re.sub(
        rf"\b(\w+)\s+NOT\s+BETWEEN\s+{value}\s+AND\s+{value}",
        r"_not_between(\1, \2, \3)",
        where,
        flags=re.IGNORECASE,
    )
    return re.sub(
        rf"\b(\w+)\s+BETWEEN\s+{value}\s+AND\s+{value}",
        r"_between(\1, \2, \3)",
        where,
        flags=re.IGNORECASE,
    )


def _normalize_nulls(where: str) -> str:
    where = re.sub(r"\b(\w+)\s+IS\s+NOT\s+NULL\b", r"_isnotnull(\1)", where, flags=re.I)
    where = re.sub(r"\b(\w+)\s+NOT\s+NULL\b", r"_isnotnull(\1)", where, flags=re.I)
    return re.sub(r"\b(\w+)\s+IS\s+NULL\b", r"_isnull(\1)", where, flags=re.I)


def _constant(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str) and node.value.upper() in {"NULL", "NONE"}:
            return None
        return node.value
    if isinstance(node, ast.Name):
        if node.id.upper() in {"NULL", "NONE"}:
            return None
        return node.id
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _constant(node.operand)
        if isinstance(value, (int, float)):
            return -value if isinstance(node.op, ast.USub) else value
    raise ValueError("Expected a constant value")


def _constants(node: ast.AST) -> list[Any]:
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_constant(element) for element in node.elts]
    return [_constant(node)]


def _to_custom_ast(node: ast.AST) -> tuple:
    if isinstance(node, ast.Expression):
        return _to_custom_ast(node.body)
    if isinstance(node, ast.BoolOp):
        name = "and" if isinstance(node.op, ast.And) else "or"
        current = _to_custom_ast(node.values[0])
        for value in node.values[1:]:
            current = (name, current, _to_custom_ast(value))
        return current
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return ("not", _to_custom_ast(node.operand))
    if isinstance(node, ast.Compare):
        if not isinstance(node.left, ast.Name) or len(node.ops) != 1:
            raise ValueError(
                "where expects a column identifier before comparison operator"
            )
        column = node.left.id
        operator = node.ops[0]
        if isinstance(operator, ast.In):
            return ("in", column, _constants(node.comparators[0]), False)
        if isinstance(operator, ast.NotIn):
            return ("in", column, _constants(node.comparators[0]), True)
        operators = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Lt: "<",
            ast.LtE: "<=",
        }
        if type(operator) not in operators:
            raise ValueError(f"Unsupported operator: {operator}")
        return (
            "cmp",
            column,
            operators[type(operator)],
            _constant(node.comparators[0]),
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if (
            name in {"_between", "_not_between"}
            and len(node.args) == 3
            and isinstance(node.args[0], ast.Name)
        ):
            return (
                "between",
                node.args[0].id,
                _constant(node.args[1]),
                _constant(node.args[2]),
                name == "_not_between",
            )
        if (
            name in {"_isnull", "_isnotnull"}
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
        ):
            return ("isnull", node.args[0].id, name == "_isnotnull")
    raise ValueError(
        "where expects a comparison operator or IN/BETWEEN/IS NULL variants after column identifier"
    )


@lru_cache(maxsize=1024)
def parse_where_expression(where: str) -> tuple:
    if not isinstance(where, str) or not where.strip():
        raise ValueError("where must be a non-empty string expression")
    try:
        source = _normalize_nulls(
            _normalize_between(
                _normalize_logical_operators(_normalize_where_syntax(where))
            )
        )
        return _to_custom_ast(ast.parse(source.strip(), mode="eval"))
    except SyntaxError as exc:
        if where.count("(") != where.count(")"):
            raise ValueError("Unbalanced parentheses in where expression") from exc
        raise ValueError("Unexpected trailing tokens in where expression") from exc


def where_columns_from_ast(tree: tuple) -> list[str]:
    result: list[str] = []

    def visit(node: tuple) -> None:
        if node[0] in {"cmp", "in", "between", "isnull"}:
            if node[1] not in result:
                result.append(node[1])
        elif node[0] in {"and", "or"}:
            visit(node[1])
            visit(node[2])
        elif node[0] == "not":
            visit(node[1])

    visit(tree)
    return result


def evaluate_where(tree: tuple, data: dict[str, Any]) -> np.ndarray:
    kind = tree[0]
    if kind == "and":
        return evaluate_where(tree[1], data) & evaluate_where(tree[2], data)
    if kind == "or":
        return evaluate_where(tree[1], data) | evaluate_where(tree[2], data)
    if kind == "not":
        return ~evaluate_where(tree[1], data)
    column = tree[1]
    if column not in data:
        raise ValueError(f"Unknown column: {column}")
    values = np.asarray(data[column])
    if kind == "cmp":
        _, _, operator, literal = tree
        if literal is None:
            if operator == "==":
                return np.asarray([value is None for value in values], dtype=bool)
            if operator == "!=":
                return np.asarray([value is not None for value in values], dtype=bool)
            raise ValueError("NULL comparisons only support == and !=")
        functions = {
            "==": np.equal,
            "!=": np.not_equal,
            ">": np.greater,
            ">=": np.greater_equal,
            "<": np.less,
            "<=": np.less_equal,
        }
        return np.asarray(functions[operator](values, literal), dtype=bool)
    if kind == "in":
        mask = np.isin(values, tree[2])
        return ~mask if tree[3] else mask
    if kind == "between":
        mask = (values >= tree[2]) & (values <= tree[3])
        return ~mask if tree[4] else mask
    if kind == "isnull":
        mask = (
            np.isnan(values)
            if np.issubdtype(values.dtype, np.floating)
            else np.asarray([value is None for value in values], dtype=bool)
        )
        return ~mask if tree[2] else mask
    raise ValueError(f"Invalid AST node: {kind}")


_parse_where_expression = parse_where_expression
_where_columns_from_ast = where_columns_from_ast
_parse_where_literal = _parse_where_literal
_tokenize_where_expression = _tokenize_where_expression
_normalize_where_syntax = _normalize_where_syntax
