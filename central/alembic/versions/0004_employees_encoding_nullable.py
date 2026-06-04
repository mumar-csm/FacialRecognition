"""employees.encoding nullable — allow biometric erasure on deactivation

Revision ID: 0004_employees_encoding_nullable
Revises: 0003_employees_pos_id
Create Date: 2026-06-04
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0004_employees_encoding_nullable"
down_revision: Union[str, None] = "0003_employees_pos_id"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Make the biometric template nullable so a deactivation can erase it
    # (encoding -> NULL) while keeping the employee row + attendance history.
    # NULL distinctly means "biometric erased / none on file", which is more
    # honest than storing empty bytes. Enrollment always supplies a non-NULL
    # value, so this only loosens the constraint for the erasure path.
    op.alter_column(
        "employees",
        "encoding",
        existing_type=sa.LargeBinary(),
        nullable=True,
    )


def downgrade() -> None:
    # Re-tighten to NOT NULL. This fails if any rows have already been erased
    # (encoding IS NULL) — backfill or hard-delete those before downgrading.
    op.alter_column(
        "employees",
        "encoding",
        existing_type=sa.LargeBinary(),
        nullable=False,
    )
