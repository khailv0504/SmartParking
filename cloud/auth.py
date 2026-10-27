from __future__ import annotations

import argparse
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import APIRouter, Form, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy import delete
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings
from .models import AdminSession

SESSION_COOKIE = "sp_session"
CSRF_COOKIE = "sp_csrf"
PASSWORD_HASHER = PasswordHasher()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def aware(value: datetime) -> datetime:
    """SQLite test DB có thể trả datetime naive; production PostgreSQL giữ timezone."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return PASSWORD_HASHER.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def create_admin_session(
    db: Session, settings: Settings, username: str, user_agent: str | None
) -> tuple[str, str]:
    now = utcnow()
    token = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(24)
    # Dọn session hết hạn khi đăng nhập, không cần thêm cron/worker.
    db.execute(delete(AdminSession).where(AdminSession.expires_at < now))
    db.add(
        AdminSession(
            token_hash=sha256(token),
            username=username,
            csrf_hash=sha256(csrf),
            created_at=now,
            expires_at=now + timedelta(hours=settings.session_hours),
            user_agent=(user_agent or "")[:500] or None,
        )
    )
    db.commit()
    return token, csrf


def resolve_admin_session(
    request: Request, factory: sessionmaker[Session]
) -> AdminSession | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    with factory() as db:
        row = db.get(AdminSession, sha256(token))
        if (
            row is None
            or row.revoked_at is not None
            or aware(row.expires_at) <= utcnow()
        ):
            return None
        # Tách object khỏi Session để middleware/page dùng sau khi connection đóng.
        db.expunge(row)
        return row


def set_auth_cookies(
    response: Response, settings: Settings, token: str, csrf: str
) -> None:
    common = dict(
        max_age=settings.session_hours * 3600,
        secure=settings.is_production,
        samesite="strict",
        path="/",
    )
    response.set_cookie(SESSION_COOKIE, token, httponly=True, **common)
    # Server render token vào form logout; JavaScript không cần đọc cookie này.
    response.set_cookie(CSRF_COOKIE, csrf, httponly=True, **common)


def clear_auth_cookies(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        SESSION_COOKIE, path="/", secure=settings.is_production, samesite="strict"
    )
    response.delete_cookie(
        CSRF_COOKIE, path="/", secure=settings.is_production, samesite="strict"
    )


def build_auth_router(settings: Settings, factory: sessionmaker[Session]) -> APIRouter:
    router = APIRouter(include_in_schema=False)

    @router.post("/auth/login")
    def login(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
    ) -> Response:
        valid_user = secrets.compare_digest(username, settings.admin_username)
        valid_password = verify_password(settings.admin_password_hash, password)
        if not (valid_user and valid_password):
            # Không nói sai username hay password để tránh dò tài khoản.
            return RedirectResponse(
                "/login?error=1", status_code=status.HTTP_303_SEE_OTHER
            )
        with factory() as db:
            token, csrf = create_admin_session(
                db, settings, username, request.headers.get("user-agent")
            )
        response = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
        set_auth_cookies(response, settings, token, csrf)
        return response

    @router.post("/auth/logout")
    def logout(request: Request, csrf_token: str = Form(...)) -> Response:
        token = request.cookies.get(SESSION_COOKIE)
        csrf_cookie = request.cookies.get(CSRF_COOKIE)
        if (
            not token
            or not csrf_cookie
            or not secrets.compare_digest(csrf_token, csrf_cookie)
        ):
            raise HTTPException(status_code=403, detail="CSRF validation failed")
        with factory() as db:
            row = db.get(AdminSession, sha256(token))
            if row is not None and secrets.compare_digest(
                row.csrf_hash, sha256(csrf_token)
            ):
                # Thu hồi phía server nên cookie cũ không thể dùng lại sau logout.
                row.revoked_at = utcnow()
                db.commit()
        response = RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
        clear_auth_cookies(response, settings)
        return response

    return router


def _cli() -> int:
    parser = argparse.ArgumentParser(description="SmartParking admin credential helper")
    parser.add_argument("command", choices=["hash-password"])
    args = parser.parse_args()
    if args.command == "hash-password":
        import getpass

        first = getpass.getpass("Admin password: ")
        second = getpass.getpass("Confirm password: ")
        if first != second:
            raise SystemExit("Passwords do not match")
        if len(first) < 12:
            raise SystemExit("Use at least 12 characters")
        print(PASSWORD_HASHER.hash(first))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
