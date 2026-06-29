from dataclasses import dataclass
from typing import Optional

import sqlparse
from sqlparse import tokens as T
from sqlparse.sql import Statement, Token, TokenList


@dataclass(frozen=True)
class SQLReadOnlyValidation:
    is_valid: bool
    error: Optional[str] = None


_MUTATING_KEYWORDS = {
    "ALTER",
    "BACKUP",
    "BEGIN",
    "COMMIT",
    "CREATE",
    "DELETE",
    "DENY",
    "DROP",
    "EXEC",
    "EXECUTE",
    "GRANT",
    "INSERT",
    "MERGE",
    "RESTORE",
    "REVOKE",
    "ROLLBACK",
    "TRUNCATE",
    "UPDATE",
    "USE",
    "WAITFOR",
}

_PASS_THROUGH_FUNCTIONS = {
    "OPENDATASOURCE",
    "OPENQUERY",
    "OPENROWSET",
}


def validate_single_read_only_select(query: str) -> SQLReadOnlyValidation:
    if not query or not query.strip():
        return SQLReadOnlyValidation(False, "Query is empty.")

    statements = sqlparse.parse(query)
    if not statements:
        return SQLReadOnlyValidation(False, "Query could not be parsed.")

    if len(statements) != 1:
        return SQLReadOnlyValidation(
            False,
            "Only one SQL statement is allowed.",
        )

    statement = statements[0]
    first_token = _first_significant_token(statement)
    if not first_token or first_token.normalized.upper() not in {"SELECT", "WITH"}:
        return SQLReadOnlyValidation(False, "Only SELECT statements are allowed.")

    if statement.get_type() != "SELECT":
        return SQLReadOnlyValidation(False, "Only SELECT statements are allowed.")

    pass_through_function = _find_pass_through_function(statement)
    if pass_through_function:
        return SQLReadOnlyValidation(
            False,
            f"Function '{pass_through_function}' is not allowed in NL2SQL queries.",
        )

    mutating_keyword = _find_mutating_keyword(statement)
    if mutating_keyword:
        return SQLReadOnlyValidation(
            False,
            f"Keyword '{mutating_keyword}' is not allowed in NL2SQL queries.",
        )

    if _contains_select_into(statement):
        return SQLReadOnlyValidation(
            False,
            "SELECT INTO statements are not allowed.",
        )

    return SQLReadOnlyValidation(True)


def _first_significant_token(statement: Statement) -> Optional[Token]:
    for token in statement.tokens:
        if _is_ignorable_token(token):
            continue
        return token
    return None


def _find_mutating_keyword(token_list: TokenList) -> Optional[str]:
    for token in token_list.flatten():
        if _is_ignorable_token(token):
            continue
        keyword = token.normalized.upper()
        if keyword in _MUTATING_KEYWORDS:
            return keyword
    return None


def _find_pass_through_function(token_list: TokenList) -> Optional[str]:
    for token in token_list.flatten():
        if _is_ignorable_token(token):
            continue
        function = token.normalized.upper()
        if function in _PASS_THROUGH_FUNCTIONS:
            return function
    return None


def _contains_select_into(token_list: TokenList) -> bool:
    for token in token_list.flatten():
        if _is_ignorable_token(token):
            continue
        if token.normalized.upper() == "INTO" and token.ttype in T.Keyword:
            return True
    return False


def _is_ignorable_token(token: Token) -> bool:
    if token.is_whitespace or token.ttype in T.Comment:
        return True

    value = str(token).lstrip()
    return value.startswith("--") or value.startswith("/*")
