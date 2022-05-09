from enum import Enum
from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
)
from sqlalchemy import select

from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship

meta = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)
Base = declarative_base(metadata=meta)

cashflows_tags_table = Table(
    "cashflows_tags",
    Base.metadata,
    Column("cashflow_id", ForeignKey("cashflows.id"), primary_key=True),
    Column("tag_id", ForeignKey("tags.id"), primary_key=True),
)


class MainCategory(Base):
    __tablename__ = "main_categories"

    id = Column(Integer, primary_key=True)
    category = Column(String, nullable=False, unique=True)

    cashflows = relationship("Cashflow", back_populates="main_category")

    def __repr__(self):
        return f"MainCategory(id={self.id!r}, category={self.category!r})"


class SubCategory(Base):
    __tablename__ = "sub_categories"

    id = Column(Integer, primary_key=True)
    category = Column(String, nullable=False, unique=True)

    cashflows = relationship("Cashflow", back_populates="sub_category")

    def __repr__(self):
        return f"SubCategory(id={self.id!r}, category={self.category!r})"


class TimeCategory(Base):
    __tablename__ = "time_categories"

    id = Column(Integer, primary_key=True)
    category = Column(String, nullable=False, unique=True)

    cashflows = relationship("Cashflow", back_populates="time_category")


class Tag(Base):
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True)
    tag = Column(String, nullable=False, unique=True)

    def __repr__(self):
        return f"Tag(tag={self.tag!r})"


class Cashflow(Base):
    __tablename__ = "cashflows"
    _unique_check_dict = {
        "main_category": {
            "class": MainCategory,
            "field": MainCategory.category,
            "field_name": "category",
        },
        "sub_category": {
            "class": SubCategory,
            "field": SubCategory.category,
            "field_name": "category",
        },
        "time_category": {
            "class": TimeCategory,
            "field": TimeCategory.category,
            "field_name": "category",
        },
        "tags": {"class": Tag, "field": Tag.tag, "field_name": "tag"},
    }

    id = Column(Integer, primary_key=True)

    date = Column(DateTime, nullable=False)
    amount = Column(Float, nullable=False)
    description = Column(String, nullable=False)

    ### Many to One ###
    main_category = relationship("MainCategory", back_populates="cashflows")
    main_category_id = Column(Integer, ForeignKey("main_categories.id"))

    sub_category = relationship("SubCategory", back_populates="cashflows")
    sub_category_id = Column(Integer, ForeignKey("sub_categories.id"))

    time_category = relationship("TimeCategory", back_populates="cashflows")
    time_category_id = Column(Integer, ForeignKey("time_categories.id"))
    #######

    ### Many to Many ###
    tags = relationship("Tag", secondary=cashflows_tags_table)
    #######

    @staticmethod
    def project_element(session, input_obj, target_class):
        chck_dict = Cashflow._unique_check_dict[target_class]

        sel_filter = Cashflow.get_filter(input_obj, chck_dict)

        if isinstance(sel_filter, list):
            existent_elements = (
                session.query(chck_dict["class"])
                .filter(chck_dict["field"].in_(sel_filter))
                .all()
            )
            if len(existent_elements) == len(sel_filter):
                return existent_elements
            else:
                new_list = []
                new_list.extend(existent_elements)
                allo = [
                    getattr(jjj, chck_dict["field_name"]) for jjj in existent_elements
                ]
                for eee in sel_filter:
                    if not (eee in allo):
                        init_dict = {chck_dict["field_name"]: eee}
                        new_list.append(chck_dict["class"](**init_dict))
                return new_list
        else:
            existent_elements = (
                session.query(chck_dict["class"])
                .filter(chck_dict["field"] == sel_filter)
                .all()
            )
            if len(existent_elements) == 1:
                return existent_elements[0]
            elif len(existent_elements) == 0:
                init_dict = {chck_dict["field_name"]: sel_filter}
                return chck_dict["class"](**init_dict)
            else:
                raise Exception("here")

    @staticmethod
    def get_filter(input_obj, comparison_dict):
        sel_filter = None
        if isinstance(input_obj, comparison_dict["class"]):
            sel_filter = getattr(input_obj, comparison_dict["field_name"])
        elif isinstance(input_obj, str):
            sel_filter = input_obj
        elif isinstance(input_obj, list):
            sel_filter = []
            for eee in input_obj:
                sel_filter.append(
                    Cashflow.get_filter(eee, comparison_dict=comparison_dict)
                )
            sel_filter = list(set(sel_filter))
        elif isinstance(input_obj, Enum):
            sel_filter = input_obj.name
        else:
            raise Exception(
                f"Impossible to find filter for object {input_obj} with comparison dictionary {comparison_dict}"
            )

        return sel_filter

    def __init__(self, **kwargs) -> None:

        session = kwargs.pop("session")
        for kkk, vvv in Cashflow._unique_check_dict.items():
            if kkk in kwargs:
                try:
                    kwargs[kkk] = Cashflow.project_element(session, kwargs[kkk], kkk)
                except Exception as excp:
                    print(
                        f"problem during projection of item {kkk} with content {kwargs[kkk]}"
                    )
                    raise excp

        super().__init__(**kwargs)

    def __repr__(self):
        return f"Cashflow(date={self.date!r}, amount={self.amount!r})"
