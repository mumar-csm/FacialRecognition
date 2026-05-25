"""employees.pos_employee_id — Oracle POS identifier per employee per store

Revision ID: 0003_employees_pos_id
Revises: 0002_sync_audit
Create Date: 2026-05-25
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0003_employees_pos_id"
down_revision: Union[str, None] = "0002_sync_audit"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "employees",
        sa.Column("pos_employee_id", sa.Text(), nullable=True),
    )
    # Partial unique index — one POS ID per store, NULLs unconstrained so
    # pre-migration rows are unaffected.
    op.create_index(
        "uniq_employees_store_pos",
        "employees",
        ["store_id", "pos_employee_id"],
        unique=True,
        postgresql_where=sa.text("pos_employee_id IS NOT NULL"),
    )
    # Format check — exactly 7 digits. NULL-tolerant so pre-migration rows pass.
    # Defense-in-depth alongside the kiosk's API-layer regex in EnrollRequest.
    op.create_check_constraint(
        "ck_employees_pos_employee_id_format",
        "employees",
        r"pos_employee_id IS NULL OR pos_employee_id ~ '^\d{7}$'",
    )

    # Reject NULL pos_employee_id on new inserts. The CHECK above tolerates NULL
    # for pre-migration rows; this trigger ensures *new* rows always have a POS
    # ID, even if a future endpoint or admin script bypasses the kiosk API.
    # UPDATEs don't fire — existing NULL rows stay untouched.
    op.execute("""
        CREATE FUNCTION reject_null_pos_employee_id_on_insert()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW.pos_employee_id IS NULL THEN
                RAISE EXCEPTION 'pos_employee_id is required for new employees';
            END IF;
            RETURN NEW;
        END;
        $$;
    """)
    op.execute("""
        CREATE TRIGGER trg_employees_pos_id_required_on_insert
        BEFORE INSERT ON employees
        FOR EACH ROW
        EXECUTE FUNCTION reject_null_pos_employee_id_on_insert();
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_employees_pos_id_required_on_insert ON employees")
    op.execute("DROP FUNCTION IF EXISTS reject_null_pos_employee_id_on_insert()")
    op.drop_constraint(
        "ck_employees_pos_employee_id_format", "employees", type_="check"
    )
    op.drop_index("uniq_employees_store_pos", table_name="employees")
    op.drop_column("employees", "pos_employee_id")
