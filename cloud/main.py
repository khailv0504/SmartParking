from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from .api import build_api_router
from .auth import build_auth_router, resolve_admin_session
from .config import Settings, get_settings
from .dashboard import register_dashboard
from .database import make_engine, make_session_factory

PROTECTED_PAGES = {"/", "/history", "/inside", "/rfid", "/device", "/plates"}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    engine = make_engine(settings)
    factory = make_session_factory(engine)

    # Docs/schema tắt ở production để không quảng bá API nội bộ.
    docs_enabled = not settings.is_production
    app = FastAPI(
        title="SmartParking Cloud",
        docs_url="/docs" if docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = factory

    @app.middleware("http")
    async def protect_admin_pages(request: Request, call_next):
        if request.url.path in PROTECTED_PAGES:
            try:
                admin = resolve_admin_session(request, factory)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Authentication database unavailable"
                )
                return JSONResponse(
                    {"detail": "Authentication service temporarily unavailable"},
                    status_code=503,
                )
            if admin is None:
                return RedirectResponse("/login", status_code=303)
            request.state.admin_session = admin
        return await call_next(request)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.url.path in PROTECTED_PAGES or request.url.path == "/login":
            response.headers["Cache-Control"] = "no-store"
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    app.include_router(build_api_router(settings, factory))
    app.include_router(build_auth_router(settings, factory))
    register_dashboard(app, settings, factory)
    return app


app = create_app()
