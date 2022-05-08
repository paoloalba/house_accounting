import os
import shutil

import pandas as pd
import numpy as np

from datetime import date, datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from locale import setlocale, LC_ALL, locale_alias, resetlocale
from contextlib import contextmanager
from sqlalchemy.sql import select

from house_accounting.models import (
    Base,
    Cashflow,
    MainCategory,
    SubCategory,
    TimeCategory,
    Tag,
)
from house_accounting.enumerators import MainCategory as EnumMainCategory
from house_accounting.enumerators import SubCategory as EnumSubCategory
from house_accounting.enumerators import TimeCategory as EnumTimeCategory


@contextmanager
def override_locale(locale_string):
    setlocale(LC_ALL, locale_alias[locale_string])
    yield
    resetlocale(LC_ALL)


class AccountingTable:
    def __init__(self, db_path) -> None:
        self.db_path = db_path

        self.db_engine = self.get_db_engine(future=False)

    # region Helper methods
    def get_db_engine(self, future=True):
        engine = create_engine(
            f"sqlite+pysqlite:///{self.db_path}", echo=False, future=future
        )
        Base.metadata.create_all(engine)
        return engine

    @staticmethod
    def parse_amount(input_str):
        return float(input_str.replace(".", "").replace(",", "."))

    @staticmethod
    def parse_date(input_str):
        try:
            return datetime.strptime(input_str, "%d %b '%y")
        except:
            return datetime.strptime(input_str, "%d %b %y")

    # endregion

    def get_df(
        self, aggregate_tags=True, normalised_by_tags=False, add_amount_sign=True
    ):
        df = pd.read_sql(
            select(Cashflow, MainCategory, SubCategory, TimeCategory, Tag)
            .join(Cashflow.main_category)
            .join(Cashflow.sub_category)
            .join(Cashflow.time_category)
            .join(Tag, Cashflow.tags),
            self.db_engine,
        )

        cols_to_drop = []
        cols_to_drop.append("main_category_id")
        cols_to_drop.append("sub_category_id")
        cols_to_drop.append("time_category_id")
        cols_to_drop.append("id_1")
        cols_to_drop.append("id_2")
        cols_to_drop.append("id_3")
        cols_to_drop.append("id_4")
        df.drop(cols_to_drop, inplace=True, axis=1)

        if aggregate_tags or normalised_by_tags:
            unique_cols = df.columns.tolist()
            unique_cols.remove("tag")

            row_list = []
            for kkk, ddd in df.groupby("id"):
                if aggregate_tags:
                    singleton = ddd.drop_duplicates(subset=unique_cols)
                    if len(singleton.index) != 1:
                        raise Exception("there are some unexpected duplicates")
                    singleton.loc[singleton.iloc[0].name, "tag"] = ";".join(
                        sorted(ddd.tag.tolist())
                    )
                    row_list.append(singleton)
                else:
                    ddd["amount"] = ddd["amount"].apply(lambda x: x / len(ddd.index))
                    row_list.append(ddd)
            df = pd.concat(row_list)

        df.sort_values("date", inplace=True, axis=0, ignore_index=True, ascending=False)
        df.rename(
            columns={
                "category": "main_category",
                "category_1": "sub_category",
                "category_2": "time_category",
            },
            inplace=True,
        )

        # if add_amount_sign:
        #     df["amount"] = df.apply(
        #         lambda x: x.amount
        #         if x.main_category == EnumMainCategory.Income.name
        #         else -x.amount,
        #         axis=1,
        #     )

        return df

    def overwrite_db_with_df(self, input_df):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        self.db_engine = self.get_db_engine(future=False)

        with Session(self.db_engine) as session:
            for ido, eee in input_df.iterrows():
                cfl_entry = Cashflow(
                    session=session,
                    date=eee.date,
                    amount=eee.amount,
                    description=eee.description,
                    main_category=eee.main_category,
                    sub_category=eee.sub_category,
                    tags=eee.tag.split(";"),
                )
                session.add(cfl_entry)

            session.commit()

    def create_backup(self):
        base_name = os.path.basename(self.db_path)
        base_name, extension = os.path.splitext(base_name)
        dir_name = os.path.dirname(self.db_path)

        new_name = os.path.join(
            dir_name, f"{base_name}-backup_{datetime.now():%Y%m%d%H%M}{extension}"
        )
        shutil.copy(self.db_path, new_name)

    # region generate default
    def generate_default_db(
        self,
    ):
        self.db_engine = self.get_db_engine(future=False)

        num_years = 2
        end_date = date.today().replace(day=27)
        start_date = end_date.replace(year=end_date.year - num_years)

        with Session(self.db_engine) as session:
            cfl_entry = Cashflow(
                session=session,
                date=start_date,
                amount=15000,
                description="initial cashflow",
                main_category=EnumMainCategory.Income.name,
                sub_category=EnumSubCategory.Initial.name,
                time_category=EnumTimeCategory.Exceptional.name,
                tags=["initial"],
            )
            session.add(cfl_entry)

            main_salary = 1800
            ref_date = start_date + pd.DateOffset(months=1)
            while ref_date <= end_date:
                cfl_entry = Cashflow(
                    session=session,
                    date=ref_date,
                    amount=main_salary,
                    description=f"salary for {ref_date:%b %y}",
                    main_category=EnumMainCategory.Income.name,
                    sub_category=EnumSubCategory.Salary.name,
                    time_category=EnumTimeCategory.Recurring.name,
                    tags=["user_1"],
                )
                session.add(cfl_entry)
                cfl_entry = Cashflow(
                    session=session,
                    date=ref_date,
                    amount=main_salary * 0.85,
                    description=f"salary for {ref_date:%b %y}",
                    main_category=EnumMainCategory.Income.name,
                    sub_category=EnumSubCategory.Salary.name,
                    time_category=EnumTimeCategory.Recurring.name,
                    tags=["user_2"],
                )
                session.add(cfl_entry)

                cfl_entry = Cashflow(
                    session=session,
                    date=ref_date,
                    amount=main_salary * 0.45,
                    description=f"rent for {ref_date:%b %y}",
                    main_category=EnumMainCategory.Outcome.name,
                    sub_category=EnumSubCategory.House.name,
                    time_category=EnumTimeCategory.Recurring.name,
                    tags=["rent"],
                )
                session.add(cfl_entry)

                cfl_entry = Cashflow(
                    session=session,
                    date=ref_date,
                    amount=main_salary * 0.03,
                    description=f"bill 1 for {ref_date:%b %y}",
                    main_category=EnumMainCategory.Outcome.name,
                    sub_category=EnumSubCategory.House.name,
                    time_category=EnumTimeCategory.Recurring.name,
                    tags=["bill_1"],
                )
                session.add(cfl_entry)
                cfl_entry = Cashflow(
                    session=session,
                    date=ref_date,
                    amount=main_salary * 0.04,
                    description=f"bill 2 for {ref_date:%b %y}",
                    main_category=EnumMainCategory.Outcome.name,
                    sub_category=EnumSubCategory.House.name,
                    time_category=EnumTimeCategory.Recurring.name,
                    tags=["bill_2"],
                )
                session.add(cfl_entry)

                ref_date += pd.DateOffset(months=1)

            ref_date = start_date + pd.offsets.Week(weekday=5)
            while ref_date <= end_date:
                cfl_entry = Cashflow(
                    session=session,
                    date=ref_date,
                    amount=main_salary * 0.051,
                    description=f"food for {ref_date:%W %b %y}",
                    main_category=EnumMainCategory.Outcome.name,
                    sub_category=EnumSubCategory.Food.name,
                    time_category=EnumTimeCategory.Recurring.name,
                    tags=["food"],
                )
                session.add(cfl_entry)

                period_start = (
                    ref_date
                    + pd.offsets.Week(weekday=0)
                    - pd.offsets.Week(1, weekday=0)
                ).floor("d")
                period_end = (ref_date + pd.offsets.Week(weekday=6)).floor("d")

                all_dates = pd.date_range(
                    start=period_start, end=period_end
                ).to_pydatetime()
                rec_prob = 0.97
                excp_prob = 0.005
                onetime_prob = 1 - rec_prob - excp_prob
                sub_cat_choices = list(EnumSubCategory)
                sub_cat_choices.remove(EnumSubCategory.NA)
                sub_cat_choices.remove(EnumSubCategory.Initial)
                for iii in range(np.random.randint(2, 10)):
                    sel_cashflow_dir = np.random.choice(
                        [EnumMainCategory.Income, EnumMainCategory.Outcome],
                        p=[0.025, 0.975],
                    )
                    sel_sub_cat = np.random.choice(sub_cat_choices)

                    sel_time_cat = np.random.choice(
                        [
                            EnumTimeCategory.Exceptional,
                            EnumTimeCategory.OneTime,
                            EnumTimeCategory.Recurring,
                        ],
                        p=[excp_prob, onetime_prob, rec_prob],
                    )

                    sel_date = np.random.choice(all_dates)
                    sel_value = np.random.uniform(low=0.01, high=10.0)
                    if sel_time_cat == EnumTimeCategory.Exceptional:
                        sel_value *= 1000
                    elif sel_time_cat == EnumTimeCategory.OneTime:
                        sel_value *= 100
                    elif sel_time_cat == EnumTimeCategory.Recurring:
                        sel_value *= 10
                    else:
                        raise Exception(f"Unrecognised time category: {sel_time_cat}")

                    if sel_sub_cat == EnumSubCategory.Salary:
                        sel_cashflow_dir = EnumMainCategory.Income

                    cfl_entry = Cashflow(
                        session=session,
                        date=sel_date,
                        amount=sel_value,
                        description=f"some random {sel_time_cat.name.lower()} cashflow for {sel_sub_cat.name.lower()}",
                        main_category=sel_cashflow_dir.name,
                        sub_category=sel_sub_cat.name,
                        time_category=sel_time_cat.name,
                        tags=["random"],
                    )
                    session.add(cfl_entry)

                ref_date += pd.DateOffset(weeks=1)

            session.commit()

    # endregion
