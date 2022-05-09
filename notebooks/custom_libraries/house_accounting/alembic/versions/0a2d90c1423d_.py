"""empty message

Revision ID: 0a2d90c1423d
Revises: 
Create Date: 2022-05-08 11:58:54.759703

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0a2d90c1423d"
down_revision = None
branch_labels = None
depends_on = None

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import Cast, type_coerce
from sqlalchemy import Date, DateTime, String


@compiles(Cast)
def _sqlite_cast(element, compiler, **kw):
    if isinstance(element.clause.type, Date) and isinstance(element.type, DateTime):
        return compiler.process(
            type_coerce(element.clause, String) + " 00:00:00.000000", **kw
        )
    else:
        return compiler.visit_cast(element, **kw)


def upgrade():
    with op.batch_alter_table("cashflows", schema=None) as batch_op:
        batch_op.alter_column(
            "date",
            existing_type=sa.DATE(),
            type_=sa.DateTime(),
            existing_nullable=False,
        )


def downgrade():
    with op.batch_alter_table("cashflows", schema=None) as batch_op:
        batch_op.alter_column(
            "date",
            existing_type=sa.DateTime(),
            type_=sa.DATE(),
            existing_nullable=False,
        )
