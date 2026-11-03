"""Private evidence image metadata (actual JPGs are stored in R2).

Revision ID: 20261103_0003
Revises: 20261102_0002
"""
from alembic import op
import sqlalchemy as sa

revision = '20261103_0003'
down_revision = '20261102_0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'evidence_images',
        sa.Column('event_id', sa.String(length=64), sa.ForeignKey('edge_events.event_id'), primary_key=True),
        sa.Column('kind', sa.String(length=16), primary_key=True),
        sa.Column('object_key', sa.String(length=256), nullable=False),
        sa.Column('sha256', sa.String(length=64), nullable=False),
        sa.Column('size_bytes', sa.Integer(), nullable=False),
        sa.Column('uploaded_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("kind IN ('annotated','plate')", name='ck_evidence_kind'),
    )


def downgrade() -> None:
    op.drop_table('evidence_images')
