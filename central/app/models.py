"""SQLAlchemy Core table definitions for the central tier.

Schema mirrors the kiosk's SQLite tables (kiosk_server.py:init_kiosk_db) but
multi-tenant by store_id and with idempotency-by-event_uuid on every sync target.

No ORM mapping — Core tables only. Alembic imports `metadata` from this module.
"""

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID


metadata = MetaData()


devices = Table(
    "devices",
    metadata,
    Column("device_id", Text, primary_key=True),
    Column("store_id", Text, nullable=False),
    Column("api_key_hash", Text, nullable=False, unique=True),
    Column("is_active", Boolean, nullable=False, server_default="true"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("last_seen_at", DateTime(timezone=True)),
)

Index("idx_devices_store", devices.c.store_id)


employees = Table(
    "employees",
    metadata,
    Column("id", Text, nullable=False),
    Column("store_id", Text, nullable=False),
    Column("display_name", Text, nullable=False),
    Column("enrolled_at", DateTime(timezone=True), nullable=False),
    Column("is_active", Boolean, nullable=False, server_default="true"),
    Column("embedder_type", Text, nullable=False),
    Column("embedding_dim", Integer, nullable=False),
    Column("encoding", LargeBinary, nullable=False),
    Column("photo", LargeBinary),
    Column("version", BigInteger, nullable=False, server_default="1"),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    PrimaryKeyConstraint("id", "store_id", name="pk_employees"),
)

Index("idx_employees_store_version", employees.c.store_id, employees.c.version)


attendance = Table(
    "attendance",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("event_uuid", UUID(as_uuid=False), nullable=False, unique=True),
    Column("store_id", Text, nullable=False),
    Column("device_id", Text, nullable=False),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("employee_id", Text, nullable=False),
    Column("distance", Float, nullable=False),
    Column("is_clock_in", Boolean, nullable=False),
    Column("camera_id", Text, nullable=False),
    Column("server_received_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    ForeignKeyConstraint(
        ["employee_id", "store_id"],
        ["employees.id", "employees.store_id"],
        name="fk_attendance_employee",
    ),
)

Index("idx_attendance_store_timestamp", attendance.c.store_id, attendance.c.timestamp)
Index("idx_attendance_employee", attendance.c.employee_id, attendance.c.store_id)


spoof_attempts = Table(
    "spoof_attempts",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("event_uuid", UUID(as_uuid=False), nullable=False, unique=True),
    Column("store_id", Text, nullable=False),
    Column("device_id", Text, nullable=False),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("camera_id", Text, nullable=False),
    Column("spoof_score", Float, nullable=False),
    Column("server_received_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

Index("idx_spoof_store_timestamp", spoof_attempts.c.store_id, spoof_attempts.c.timestamp)
