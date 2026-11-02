"""Add RFID card registry snapshots (no changes to existing events).

Revision ID: 20261102_0002
Revises: 20261027_0001
"""
from alembic import op
import sqlalchemy as sa

revision = '20261102_0002'
down_revision = '20261027_0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'rfid_snapshot_state',
        sa.Column('device_id', sa.String(length=80), primary_key=True),
        sa.Column('snapshot_version', sa.Integer(), nullable=False),
        sa.Column('payload_hash', sa.String(length=64), nullable=False),
        sa.Column('received_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        'rfid_cards',
        sa.Column('device_id', sa.String(length=80), primary_key=True),
        sa.Column('uid', sa.String(length=80), primary_key=True),
        sa.Column('enabled', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('synced_at', sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('rfid_cards')
    op.drop_table('rfid_snapshot_state')
