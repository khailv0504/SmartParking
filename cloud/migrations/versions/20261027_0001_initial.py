"""Initial SmartParking Cloud schema.

Revision ID: 20261027_0001
Revises:
Create Date: 2026-10-27
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261027_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "edge_events",
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("device_id", sa.String(length=80), nullable=False),
        sa.Column("uid", sa.String(length=80), nullable=True),
        sa.Column("plate", sa.String(length=32), nullable=True),
        sa.Column("event_type", sa.String(length=12), nullable=False),
        sa.Column("reason", sa.String(length=160), nullable=True),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("event_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ocr_confidence", sa.Float(), nullable=True),
        sa.Column("detection_confidence", sa.Float(), nullable=True),
        sa.Column("iqa_score", sa.Float(), nullable=True),
        sa.Column("inference_ms", sa.Float(), nullable=True),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.CheckConstraint(
            "event_type IN ('ENTRY','EXIT','DENY','REVIEW','ERROR')",
            name="ck_edge_events_type",
        ),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index("ix_edge_events_timestamp", "edge_events", ["event_timestamp"])
    op.create_index("ix_edge_events_plate", "edge_events", ["plate"])
    op.create_index("ix_edge_events_uid", "edge_events", ["uid"])
    op.create_index("ix_edge_events_type", "edge_events", ["event_type"])
    op.create_index("ix_edge_events_session", "edge_events", ["session_id"])

    op.create_table(
        "parking_sessions",
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("entry_event_id", sa.String(length=64), nullable=True),
        sa.Column("exit_event_id", sa.String(length=64), nullable=True),
        sa.Column("uid", sa.String(length=80), nullable=True),
        sa.Column("plate", sa.String(length=32), nullable=True),
        sa.Column("entry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("exit_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["entry_event_id"], ["edge_events.event_id"]),
        sa.ForeignKeyConstraint(["exit_event_id"], ["edge_events.event_id"]),
        sa.PrimaryKeyConstraint("session_id"),
        sa.UniqueConstraint("entry_event_id"),
        sa.UniqueConstraint("exit_event_id"),
    )
    op.create_index(
        "ix_parking_sessions_open", "parking_sessions", ["exit_at", "entry_at"]
    )
    op.create_index("ix_parking_sessions_plate", "parking_sessions", ["plate"])
    op.create_index("ix_parking_sessions_uid", "parking_sessions", ["uid"])

    op.create_table(
        "admin_sessions",
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("username", sa.String(length=80), nullable=False),
        sa.Column("csrf_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("token_hash"),
    )
    op.create_index("ix_admin_sessions_expires_at", "admin_sessions", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_admin_sessions_expires_at", table_name="admin_sessions")
    op.drop_table("admin_sessions")
    op.drop_index("ix_parking_sessions_uid", table_name="parking_sessions")
    op.drop_index("ix_parking_sessions_plate", table_name="parking_sessions")
    op.drop_index("ix_parking_sessions_open", table_name="parking_sessions")
    op.drop_table("parking_sessions")
    op.drop_index("ix_edge_events_session", table_name="edge_events")
    op.drop_index("ix_edge_events_type", table_name="edge_events")
    op.drop_index("ix_edge_events_uid", table_name="edge_events")
    op.drop_index("ix_edge_events_plate", table_name="edge_events")
    op.drop_index("ix_edge_events_timestamp", table_name="edge_events")
    op.drop_table("edge_events")
