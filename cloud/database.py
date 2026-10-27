from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import Settings


class Base(DeclarativeBase):
    pass


def make_engine(settings: Settings) -> Engine:
    kwargs: dict = {"pool_pre_ping": True}
    if settings.database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # Một Web Service nhỏ không nên chiếm quá nhiều connection của Postgres Free.
        kwargs.update(
            pool_size=5,
            max_overflow=2,
            pool_recycle=300,
            connect_args={"connect_timeout": 5},
        )
    return create_engine(settings.database_url, **kwargs)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


def session_dependency(
    factory: sessionmaker[Session],
) -> Generator[Session, None, None]:
    with factory() as session:
        yield session
