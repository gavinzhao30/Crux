"""Evaluate predeclared outcome predicates against structured diagnostic results."""

import ast
import operator
from collections.abc import Iterable

_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
}


def _predicate(node: ast.AST, values: dict):
    if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool, type(None)):
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_predicate(item, values) for item in node.elts]
    if isinstance(node, ast.UnaryOp):
        value = _predicate(node.operand, values)
        if isinstance(node.op, ast.Not) and isinstance(value, bool):
            return not value
        if isinstance(node.op, ast.USub) and type(value) in (int, float):
            return -value
    if isinstance(node, ast.BoolOp):
        terms = [_predicate(term, values) for term in node.values]
        if all(isinstance(term, bool) for term in terms):
            return all(terms) if isinstance(node.op, ast.And) else any(terms)
    if isinstance(node, ast.Compare):
        operands = [_predicate(item, values) for item in [node.left, *node.comparators]]
        return all(
            _OPERATORS[type(op)](left, right)
            for op, left, right in zip(node.ops, operands[:-1], operands[1:], strict=True)
        )
    raise ValueError(f"Unsupported outcome predicate: {type(node).__name__}")


def choose_bin(rules: Iterable[tuple[str, str]], result: dict) -> str:
    matched = []
    for bin_id, rule in rules:
        if len(rule) > 512:
            raise ValueError("Outcome rule exceeds the length limit")
        applies = _predicate(ast.parse(rule, mode="eval").body, result)
        if not isinstance(applies, bool):
            raise TypeError(f"Bin {bin_id!r} must evaluate to a boolean")
        if applies:
            matched.append(bin_id)
    if len(matched) != 1:
        raise ValueError(f"Expected exactly one matching outcome bin, got {len(matched)}")
    return matched[0]
