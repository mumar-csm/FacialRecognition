"""initial schema — devices, employees, attendance, spoof_attempts

Revision ID: 0001_init
Revises:
Create Date: 2026-05-18
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID


revision: str = "0001_init"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "devices",
        sa.Column("device_id", sa.Text(), primary_key=True),
        sa.Column("store_id", sa.Text(), nullable=False),
        sa.Column("api_key_hash", sa.Text(), nullable=False, unique=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("last_seen_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_devices_store", "devices", ["store_id"])

    op.create_table(
        "employees",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("store_id", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("enrolled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("embedder_type", sa.Text(), nullable=False),
        sa.Column("embedding_dim", sa.Integer(), nullable=False),
        sa.Column("encoding", sa.LargeBinary(), nullable=False),
        sa.Column("photo", sa.LargeBinary()),
        sa.Column("version", sa.BigInteger(), nullable=False, server_default=sa.text("1")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id", "store_id", name="pk_employees"),
    )
    op.create_index("idx_employees_store_version", "employees", ["store_id", "version"])

    op.create_table(
        "attendance",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("event_uuid", UUID(as_uuid=False), nullable=False, unique=True),
        sa.Column("store_id", sa.Text(), nullable=False),
        sa.Column("device_id", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("employee_id", sa.Text(), nullable=False),
        sa.Column("distance", sa.Float(), nullable=False),
        sa.Column("is_clock_in", sa.Boolean(), nullable=False),
        sa.Column("camera_id", sa.Text(), nullable=False),
        sa.Column(
            "server_received_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["employee_id", "store_id"],
            ["employees.id", "employees.store_id"],
            name="fk_attendance_employee",
        ),
    )
    op.create_index(
        "idx_attendance_store_timestamp", "attendance", ["store_id", "timestamp"]
    )
    op.create_index(
        "idx_attendance_employee", "attendance", ["employee_id", "store_id"]
    )

    op.create_table(
        "spoof_attempts",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("event_uuid", UUID(as_uuid=False), nullable=False, unique=True),
        sa.Column("store_id", sa.Text(), nullable=False),
        sa.Column("device_id", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("camera_id", sa.Text(), nullable=False),
        sa.Column("spoof_score", sa.Float(), nullable=False),
        sa.Column(
            "server_received_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "idx_spoof_store_timestamp", "spoof_attempts", ["store_id", "timestamp"]
    )


def downgrade() -> None:
    op.drop_index("idx_spoof_store_timestamp", table_name="spoof_attempts")
    op.drop_table("spoof_attempts")
    op.drop_index("idx_attendance_employee", table_name="attendance")
    op.drop_index("idx_attendance_store_timestamp", table_name="attendance")
    op.drop_table("attendance")
    op.drop_index("idx_employees_store_version", table_name="employees")
    op.drop_table("employees")
    op.drop_index("idx_devices_store", table_name="devices")
    op.drop_table("devices")
