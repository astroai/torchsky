"""WHERE expression parser delegated through torchfits when available."""

try:
    from torchfits.where import (
        evaluate_where,
        normalize_where_syntax as _normalize_where_syntax,
        parse_where_expression,
        parse_where_literal as _parse_where_literal,
        tokenize_where_expression as _tokenize_where_expression,
        where_identifier_re as _WHERE_IDENT_RE,
        where_columns_from_ast,
    )
except ModuleNotFoundError:
    from . import _where_compat as _compat

    _WHERE_IDENT_RE = _compat._WHERE_IDENT_RE
    _normalize_where_syntax = _compat._normalize_where_syntax
    _parse_where_expression = _compat._parse_where_expression
    _parse_where_literal = _compat._parse_where_literal
    _tokenize_where_expression = _compat._tokenize_where_expression
    _where_columns_from_ast = _compat._where_columns_from_ast
    evaluate_where = _compat.evaluate_where
    parse_where_expression = _compat.parse_where_expression
    where_columns_from_ast = _compat.where_columns_from_ast

_parse_where_expression = parse_where_expression
_where_columns_from_ast = where_columns_from_ast
