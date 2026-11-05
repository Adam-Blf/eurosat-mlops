"""Token authentication. Missing credentials give 401, unknown tokens give 403."""

from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

ADMIN_USER = "admin"


@dataclass(frozen=True)
class Principal:
    user: str
    is_admin: bool


def extract_token(request: Request) -> str | None:
    """Bearer token from Authorization, or X-API-Token as a fallback."""
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    return request.headers.get("x-api-token") or None


def _match(candidate: str, tokens: dict[str, str]) -> str | None:
    """Constant-time lookup: every stored token is compared, so timing leaks nothing."""
    found = None
    for token, user in tokens.items():
        if hmac.compare_digest(candidate.encode(), token.encode()):
            found = user
    return found


class Authenticator:
    def __init__(self, tokens: dict[str, str], admin_token: str) -> None:
        self._tokens = tokens
        self._admin = admin_token

    def authenticate(self, request: Request) -> Principal:
        token = extract_token(request)
        if token is None:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "missing token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if self._admin and hmac.compare_digest(token.encode(), self._admin.encode()):
            return Principal(ADMIN_USER, True)
        user = _match(token, self._tokens)
        if user is None:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "invalid token")
        return Principal(user, False)
