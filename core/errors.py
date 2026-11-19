"""Errores de dominio (HTTP 422: la entrada del usuario no se pudo interpretar o resolver)."""


class CalcError(Exception):
    http_status = 422
