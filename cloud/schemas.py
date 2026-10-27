from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EventType(StrEnum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    DENY = "DENY"
    REVIEW = "REVIEW"
    ERROR = "ERROR"


ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")


class EdgeEventIn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    event_id: str = Field(min_length=1, max_length=64)
    device_id: str = Field(min_length=1, max_length=80)
    uid: str | None = Field(default=None, max_length=80)
    plate: str | None = Field(default=None, max_length=32)
    event_type: EventType
    reason: str | None = Field(default=None, max_length=160)
    session_id: str | None = Field(default=None, min_length=1, max_length=64)
    timestamp: datetime
    ocr_confidence: float | None = Field(default=None, ge=0, le=1)
    detection_confidence: float | None = Field(default=None, ge=0, le=1)
    iqa_score: float | None = Field(default=None, ge=0, le=1)
    inference_ms: float | None = Field(default=None, ge=0)

    @field_validator("event_id", "session_id")
    @classmethod
    def safe_identifier(cls, value: str | None) -> str | None:
        if value is not None and not ID_RE.fullmatch(value):
            raise ValueError("must contain only letters, numbers, '.', '_', ':' or '-'")
        return value

    @field_validator("uid", "plate")
    @classmethod
    def normalize_identifier(cls, value: str | None) -> str | None:
        return value.upper() if value else value

    @field_validator("timestamp")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone offset")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def valid_session_event(self) -> "EdgeEventIn":
        if self.event_type in (EventType.ENTRY, EventType.EXIT):
            missing = [
                name
                for name in ("uid", "plate", "session_id")
                if getattr(self, name) in (None, "")
            ]
            if missing:
                raise ValueError(
                    f"{self.event_type.value} requires {', '.join(missing)}"
                )
        return self

    def canonical_hash(self) -> str:
        # Canonical JSON làm cho +00:00 và Z có cùng ý nghĩa idempotency.
        payload = self.model_dump(mode="json")
        payload["timestamp"] = self.timestamp.astimezone(timezone.utc).isoformat(
            timespec="microseconds"
        )
        raw = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class EventAccepted(BaseModel):
    event_id: str
    status: str
    duplicate: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
