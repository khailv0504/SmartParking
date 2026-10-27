from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Cấu hình chỉ đọc từ environment; không chứa giá trị bí mật mặc định."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    database_url: str = Field(alias="DATABASE_URL")
    # Chấp nhận token Edge hiện hữu; tài liệu vẫn yêu cầu tạo token mới đủ dài.
    device_token: str = Field(alias="SMARTPARKING_DEVICE_TOKEN", min_length=1)
    admin_username: str = Field(alias="ADMIN_USERNAME", min_length=3, max_length=80)
    admin_password_hash: str = Field(alias="ADMIN_PASSWORD_HASH", min_length=20)
    nicegui_storage_secret: str = Field(alias="NICEGUI_STORAGE_SECRET", min_length=32)
    app_env: Literal["development", "test", "production"] = Field(
        default="production", alias="APP_ENV"
    )
    session_hours: int = Field(default=12, alias="SESSION_HOURS", ge=1, le=168)
    recent_device_minutes: int = Field(
        default=10, alias="RECENT_DEVICE_MINUTES", ge=1, le=1440
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("database_url")
    @classmethod
    def normalize_database_url(cls, value: str) -> str:
        value = value.strip()
        # Render đôi khi cung cấp scheme postgres:// cũ; SQLAlchemy cần driver rõ ràng.
        if value.startswith("postgres://"):
            return "postgresql+psycopg://" + value.removeprefix("postgres://")
        if value.startswith("postgresql://"):
            return "postgresql+psycopg://" + value.removeprefix("postgresql://")
        return value

    @field_validator(
        "device_token",
        "admin_username",
        "admin_password_hash",
        "nicegui_storage_secret",
    )
    @classmethod
    def no_surrounding_whitespace(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("must not contain leading or trailing whitespace")
        return value

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
