"""SQLAlchemy Core table definitions for the central tier.

Schema mirrors the kiosk's SQLite tables (kiosk_server.py:init_kiosk_db) but
multi-tenant by store_id and with idempotency-by-event_uuid on every sync target.

No ORM mapping — Core tables only. Alembic imports `metadata` from this module.
"""

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
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
    text,
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
    # Biometric template. Nullable because a deactivation erases it (sets it to
    # NULL) while retaining the row + attendance history — see migration 0004
    # and sync._handle_deactivation. Enrollment always supplies a value, so the
    # only path that leaves this NULL is a deliberate erasure.
    Column("encoding", LargeBinary),
    Column("photo", LargeBinary),
    # Oracle POS employee identifier — used by the Oracle push worker to map
    # attendance rows to Simphony employees. Nullable for rows enrolled before
    # this column existed; new enrollments require it at the kiosk API layer.
    Column("pos_employee_id", Text),
    # Store-monotonic change counter for the roster watermark. Backed by the
    # `employee_change_seq` sequence (migration 0006) — every enrollment/
    # deactivation assigns a fresh nextval, so a store's versions are unique and
    # strictly increasing. Server default fires for inserts that omit it; the
    # sync handlers set it explicitly on the update paths.
    Column(
        "version",
        BigInteger,
        nullable=False,
        server_default=text("nextval('employee_change_seq')"),
    ),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    PrimaryKeyConstraint("id", "store_id", name="pk_employees"),
    CheckConstraint(
        r"pos_employee_id IS NULL OR pos_employee_id ~ '^\d{7}$'",
        name="ck_employees_pos_employee_id_format",
    ),
)

Index("idx_employees_store_version", employees.c.store_id, employees.c.version)
# Partial unique index: one POS ID per store. Multiple NULLs allowed so
# pre-migration rows aren't constrained.
Index(
    "uniq_employees_store_pos",
    employees.c.store_id,
    employees.c.pos_employee_id,
    unique=True,
    postgresql_where=employees.c.pos_employee_id.isnot(None),
)


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


# Audit trail for events the batch endpoint accepted at the HTTP layer but
# could not insert (store_id mismatch, unknown kind, malformed payload, etc.).
# The kiosk treats a 2xx as "drop from outbox" so without this table those
# events would only exist in stdout logs. Keeping it loose on purpose:
# device_id has no FK to devices so we can later log auth failures here too,
# and event_uuid is TEXT (not UUID) so malformed UUIDs are still capturable.
sync_audit = Table(
    "sync_audit",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("received_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("device_id", Text, nullable=False),
    Column("store_id", Text),
    Column("event_uuid", Text),
    Column("kind", Text),
    Column("reason", Text, nullable=False),
    Column("payload_preview", Text),
)

Index("idx_sync_audit_received", sync_audit.c.received_at)
Index("idx_sync_audit_device_received", sync_audit.c.device_id, sync_audit.c.received_at)
Index("idx_sync_audit_reason_received", sync_audit.c.reason, sync_audit.c.received_at)


# Durable record of *applied* biometric erasures. Written by the deactivation
# handler ONLY when the erasure actually lands (the UPDATE affected a row), so
# redelivered/stale no-op events never create phantom audit entries. This is the
# compliance trail: "employee X at store Y was erased, reported by device Z, at
# time T". Named by the action (deletion) rather than the actor so HQ-initiated
# deletes in Step 2b can write here too. Deliberately loose — no FK to employees
# — so the audit survives even if the employee row is later hard-purged (#3).
deletion_audit = Table(
    "deletion_audit",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("erased_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("device_id", Text, nullable=False),
    Column("store_id", Text, nullable=False),
    Column("employee_id", Text, nullable=False),
    Column("event_uuid", Text, nullable=False),
    # The kiosk-side timestamp from the deactivation event (when the delete was
    # issued), distinct from erased_at (when central applied it).
    Column("event_timestamp", DateTime(timezone=True), nullable=False),
)

Index("idx_deletion_audit_erased", deletion_audit.c.erased_at)
Index("idx_deletion_audit_store_employee", deletion_audit.c.store_id, deletion_audit.c.employee_id)
