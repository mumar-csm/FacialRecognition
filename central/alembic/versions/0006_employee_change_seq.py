"""employee_change_seq — store-monotonic version for the roster watermark

Revision ID: 0006_employee_change_seq
Revises: 0005_deletion_audit
Create Date: 2026-06-11

Background: employees.version was a per-row counter (each new employee started
at 1, bumped +1 on its own changes). The Step 2b roster pull uses
`WHERE version > since` as a scalar watermark, which requires version to be
monotonic across the whole store — a per-row counter repeats values between
employees, so the watermark conflates them and silently drops changes (the
second deactivation in a store collides with the watermark the first one set).

Fix: drive version from a single Postgres SEQUENCE. nextval() is atomic and
strictly increasing, so every write across every store gets a unique, ordered
value. Per-store `WHERE store_id=? AND version > since` is then correct because
a store's versions are a strictly-increasing subset of the global sequence
(cross-store gaps are irrelevant — the watermark is per store).
"""
from typing import Sequence, Union

from alembic import op


revision: str = "0006_employee_change_seq"
down_revision: Union[str, None] = "0005_deletion_audit"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. The sequence that now backs every version assignment.
    op.execute("CREATE SEQUENCE IF NOT EXISTS employee_change_seq")

    # 2. Re-stamp existing rows onto one coherent global order. Assign
    #    row_number() (1..N) ordered by updated_at (id as tiebreaker) so past
    #    changes keep their relative order, then advance the sequence past the
    #    max so the next nextval() continues cleanly. Doing the ordering in the
    #    CTE — not via nextval() inside the UPDATE — makes the result
    #    deterministic regardless of Postgres's row-update order.
    op.execute(
        """
        WITH ordered AS (
            SELECT id, store_id,
                   row_number() OVER (ORDER BY updated_at, id) AS rn
            FROM employees
        )
        UPDATE employees e
        SET version = o.rn
        FROM ordered o
        WHERE e.id = o.id AND e.store_id = o.store_id
        """
    )
    # Point the sequence just past the highest assigned value (>=1 so the empty
    # table case is valid). nextval() therefore returns max+1 on the next write.
    op.execute(
        "SELECT setval('employee_change_seq', "
        "GREATEST((SELECT COALESCE(MAX(version), 0) FROM employees), 1), true)"
    )

    # 3. New inserts that omit version now draw from the sequence too. The
    #    enrollment/deactivation UPDATE paths set it explicitly (see sync.py),
    #    but the default keeps any other inserter consistent.
    op.execute(
        "ALTER TABLE employees "
        "ALTER COLUMN version SET DEFAULT nextval('employee_change_seq')"
    )

    # 4. Tie the sequence's lifetime to the column so it's cleaned up with it.
    op.execute("ALTER SEQUENCE employee_change_seq OWNED BY employees.version")


def downgrade() -> None:
    # Revert to the old per-row default of 1. Existing version values are left
    # as-is (harmless integers); only the generation mechanism changes back.
    op.execute("ALTER TABLE employees ALTER COLUMN version SET DEFAULT 1")
    op.execute("DROP SEQUENCE IF EXISTS employee_change_seq")