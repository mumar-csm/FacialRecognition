"""sync_audit — events the batch endpoint accepted but could not insert

Revision ID: 0002_sync_audit
Revises: 0001_init
Create Date: 2026-05-19
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0002_sync_audit"
down_revision: Union[str, None] = "0001_init"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sync_audit",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("device_id", sa.Text(), nullable=False),
        sa.Column("store_id", sa.Text()),
        sa.Column("event_uuid", sa.Text()),
        sa.Column("kind", sa.Text()),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("payload_preview", sa.Text()),
    )
    op.create_index("idx_sync_audit_received", "sync_audit", ["received_at"])
    op.create_index(
        "idx_sync_audit_device_received", "sync_audit", ["device_id", "received_at"]
    )
    op.create_index(
        "idx_sync_audit_reason_received", "sync_audit", ["reason", "received_at"]
    )


def downgrade() -> None:
    op.drop_index("idx_sync_audit_reason_received", table_name="sync_audit")
    op.drop_index("idx_sync_audit_device_received", table_name="sync_audit")
    op.drop_index("idx_sync_audit_received", table_name="sync_audit")
    op.drop_table("sync_audit")
