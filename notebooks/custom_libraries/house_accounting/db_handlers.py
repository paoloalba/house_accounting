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

from house_accounting.models import Base, Cashflow, MainCategory, SubCategory, Tag
from house_accounting.enumerators import MainCategory as EnumMainCategory
from house_accounting.enumerators import SubCategory as EnumSubCategory

@contextmanager
def override_locale(locale_string):
    setlocale(LC_ALL, locale_alias[locale_string])
    yield
    resetlocale(LC_ALL)

class AccountingTable:

    def __init__(self, db_path) -> None:
        self.db_path = db_path

        self.db_engine = self.get_db_engine(future=False)

    #region Helper methods
    def get_db_engine(self, future=True):
        engine = create_engine(f"sqlite+pysqlite:///{self.db_path}", echo=False, future=future)
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
    #endregion

    def get_df(
        self,
        ):
        df = pd.read_sql(
            select(Cashflow, MainCategory, SubCategory, Tag)
            .join(Cashflow.main_category)
            .join(Cashflow.sub_category)
            .join(Tag, Cashflow.tags),
            self.db_engine,
        )

        cols_to_drop = []
        cols_to_drop.append("main_category_id")
        cols_to_drop.append("sub_category_id")
        cols_to_drop.append("id_1")
        cols_to_drop.append("id_2")
        cols_to_drop.append("id_3")
        df.drop(cols_to_drop, inplace=True, axis=1)

        unique_cols = df.columns.tolist()
        unique_cols.remove("tag")

        row_list = []
        grp_df = df.groupby("id")
        for kkk, ddd in grp_df:
            singleton = ddd.drop_duplicates(subset=unique_cols)
            if len(singleton.index) != 1:
                raise Exception("there are some unexpected duplicates")
            singleton.loc[singleton.iloc[0].name, "tag"] = ";".join(sorted(ddd.tag.tolist()))
            row_list.append(singleton)
        df = pd.concat(row_list)

        df.sort_values("date", inplace=True, axis=0, ignore_index=True)
        df.rename(columns={"category":"main_category", "category_1":"sub_category"}, inplace=True)

        return df

    def overwrite_db_with_df(
        self,
        input_df
    ):
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

        new_name = os.path.join(dir_name, f"{base_name}-backup_{datetime.now():%Y%m%d%H%M}{extension}")
        shutil.copy(self.db_path, new_name)

    #region generate default
    def generate_default_db(
        self,
    ):
        self.db_engine = self.get_db_engine(future=False)

        end_date = date.today().replace(day=27)
        start_date = end_date.replace(year=end_date.year - 2)

        all_dates = pd.date_range(start=start_date, end=end_date).to_pydatetime()

        with Session(self.db_engine) as session:
            cfl_entry = Cashflow(
                session=session,
                date=start_date,
                amount=15000,
                description="initial cashflow",
                main_category=EnumMainCategory.Income.name,
                sub_category=EnumSubCategory.Initial.name,
                tags=["initial"],
            )
            session.add(cfl_entry)

            ref_date = start_date + pd.DateOffset(months=1)
            while ref_date < end_date:
                cfl_entry = Cashflow(
                    session=session,
                    date=ref_date,
                    amount=1800,
                    description=f"salary for {ref_date:%b %y}",
                    main_category=EnumMainCategory.Income.name,
                    sub_category=EnumSubCategory.Salary.name,
                    tags=["salary"],
                )
                session.add(cfl_entry)
                ref_date += pd.DateOffset(months=1)

                prob_func = lambda x: 1/x**(1/2)
                prob_func = np.vectorize(prob_func)
                chc_arr = np.arange(0.1, 100, step=0.1)
                chc_arr = np.append(chc_arr, np.arange(100, 1000, step=10))
                chc_arr = np.append(chc_arr, np.arange(1000, 10000, step=100))

                sum_invers = np.sum(prob_func(chc_arr))
                probs=[prob_func(ppp)/sum_invers for ppp in chc_arr]

                for iii in range(np.random.randint(10)):
                    # sel_value = np.random.choice(chc_arr, p=probs)
                    sel_value = np.random.choice(chc_arr)
                    sel_date = np.random.choice(all_dates)

                    if sel_value > 1000:
                        sel_sub_cat = EnumSubCategory.Exceptional
                    elif sel_value > 100:
                        sel_sub_cat = EnumSubCategory.OneTime
                    else:
                        sel_sub_cat = EnumSubCategory.Generic

                    cfl_entry = Cashflow(
                        session=session,
                        date=sel_date,
                        amount=sel_value,
                        description=f"outcome",
                        main_category=EnumMainCategory.Outcome.name,
                        sub_category=sel_sub_cat.name,
                        tags=["outcome"],
                    )
                    session.add(cfl_entry)

            session.commit()
    #endregion
