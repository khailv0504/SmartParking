from __future__ import annotations

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class EdgeEvent(Base):
    __tablename__ = "edge_events"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ('ENTRY','EXIT','DENY','REVIEW','ERROR')",
            name="ck_edge_events_type",
        ),
        Index("ix_edge_events_timestamp", "event_timestamp"),
        Index("ix_edge_events_plate", "plate"),
        Index("ix_edge_events_uid", "uid"),
        Index("ix_edge_events_type", "event_type"),
        Index("ix_edge_events_session", "session_id"),
    )

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    device_id: Mapped[str] = mapped_column(String(80), nullable=False)
    uid: Mapped[str | None] = mapped_column(String(80))
    plate: Mapped[str | None] = mapped_column(String(32))
    event_type: Mapped[str] = mapped_column(String(12), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(160))
    session_id: Mapped[str | None] = mapped_column(String(64))
    event_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    ocr_confidence: Mapped[float | None]
    detection_confidence: Mapped[float | None]
    iqa_score: Mapped[float | None]
    inference_ms: Mapped[float | None]
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class ParkingSession(Base):
    """Projection từ ENTRY/EXIT, không phải nguồn dữ liệu gốc.

    entry_* nullable để lưu được EXIT tới trước ENTRY. Khi ENTRY tới sau, cùng
    session_id sẽ hoàn thiện bản ghi thay vì tạo phiên thứ hai.
    """

    __tablename__ = "parking_sessions"
    __table_args__ = (
        Index("ix_parking_sessions_open", "exit_at", "entry_at"),
        Index("ix_parking_sessions_plate", "plate"),
        Index("ix_parking_sessions_uid", "uid"),
    )

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    entry_event_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("edge_events.event_id"), unique=True
    )
    exit_event_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("edge_events.event_id"), unique=True
    )
    uid: Mapped[str | None] = mapped_column(String(80))
    plate: Mapped[str | None] = mapped_column(String(32))
    entry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    exit_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class AdminSession(Base):
    __tablename__ = "admin_sessions"
    __table_args__ = (Index("ix_admin_sessions_expires_at", "expires_at"),)

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(80), nullable=False)
    csrf_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    user_agent: Mapped[str | None] = mapped_column(Text)
