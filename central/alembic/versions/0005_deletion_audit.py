"""deletion_audit — durable record of applied biometric erasures

Revision ID: 0005_deletion_audit
Revises: 0004_employees_encoding_nullable
Create Date: 2026-06-05
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0005_deletion_audit"
down_revision: Union[str, None] = "0004_employees_encoding_nullable"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "deletion_audit",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "erased_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("device_id", sa.Text(), nullable=False),
        sa.Column("store_id", sa.Text(), nullable=False),
        sa.Column("employee_id", sa.Text(), nullable=False),
        sa.Column("event_uuid", sa.Text(), nullable=False),
        sa.Column("event_timestamp", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_deletion_audit_erased", "deletion_audit", ["erased_at"])
    op.create_index(
        "idx_deletion_audit_store_employee",
        "deletion_audit",
        ["store_id", "employee_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_deletion_audit_store_employee", table_name="deletion_audit")
    op.drop_index("idx_deletion_audit_erased", table_name="deletion_audit")
    op.drop_table("deletion_audit")
