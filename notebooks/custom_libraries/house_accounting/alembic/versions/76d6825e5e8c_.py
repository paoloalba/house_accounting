"""empty message

Revision ID: 76d6825e5e8c
Revises: 
Create Date: 2022-04-06 21:16:08.382316

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "76d6825e5e8c"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "time_categories",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("category", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_time_categories")),
        sa.UniqueConstraint("category", name=op.f("uq_time_categories_category")),
    )
    with op.batch_alter_table("cashflows", schema=None) as batch_op:
        batch_op.add_column(sa.Column("time_category_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            batch_op.f("fk_cashflows_time_category_id_time_categories"),
            "time_categories",
            ["time_category_id"],
            ["id"],
        )

    with op.batch_alter_table("main_categories", schema=None) as batch_op:
        batch_op.create_unique_constraint(
            batch_op.f("uq_main_categories_category"), ["category"]
        )

    with op.batch_alter_table("sub_categories", schema=None) as batch_op:
        batch_op.create_unique_constraint(
            batch_op.f("uq_sub_categories_category"), ["category"]
        )

    with op.batch_alter_table("tags", schema=None) as batch_op:
        batch_op.create_unique_constraint(batch_op.f("uq_tags_tag"), ["tag"])

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("tags", schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f("uq_tags_tag"), type_="unique")

    with op.batch_alter_table("sub_categories", schema=None) as batch_op:
        batch_op.drop_constraint(
            batch_op.f("uq_sub_categories_category"), type_="unique"
        )

    with op.batch_alter_table("main_categories", schema=None) as batch_op:
        batch_op.drop_constraint(
            batch_op.f("uq_main_categories_category"), type_="unique"
        )

    with op.batch_alter_table("cashflows", schema=None) as batch_op:
        batch_op.drop_constraint(
            batch_op.f("fk_cashflows_time_category_id_time_categories"),
            type_="foreignkey",
        )
        batch_op.drop_column("time_category_id")

    op.drop_table("time_categories")
    # ### end Alembic commands ###
