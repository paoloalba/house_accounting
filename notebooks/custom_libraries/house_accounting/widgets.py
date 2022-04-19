import numpy as np
import pandas as pd
import ipywidgets as widgets
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

from IPython.display import display
from datetime import date, timedelta
from sqlalchemy.sql import select, delete
from sqlalchemy.orm import Session
from pandas.tseries.offsets import MonthEnd, Week

from house_accounting.models import (
    Cashflow,
    MainCategory,
    SubCategory,
    TimeCategory,
    Tag,
)
from house_accounting.enumerators import MainCategory as EnumMainCat
from house_accounting.enumerators import SubCategory as EnumSubCat
from house_accounting.enumerators import TimeCategory as EnumTimeCat
from house_accounting.enumerators import SampleFrequency


class WidgetBase:
    def __init__(self, base_dir) -> None:
        self.main_element = None
        self.base_dir = base_dir

    def display(self):
        display(self.main_element)


class AccountingDBManager(WidgetBase):
    tags_sep = ";"

    def __init__(self, base_dir, acc_table) -> None:
        super().__init__(base_dir)

        self.acc_table = acc_table
        self.db_engine = acc_table.db_engine

        left_box_list = []
        right_box_list = []

        button_layout = widgets.Layout(width="auto", height="auto")

        # region Left Box
        clear_fields_button = widgets.Button(
            description="Clear all fields",
            disabled=False,
            button_style="",
            tooltip="",
            icon="check",
            layout=button_layout,
        )
        left_box_list.append(clear_fields_button)
        left_box_list.append(
            widgets.HTML(
                value="<hr>",
            )
        )

        df = pd.read_sql(
            select(MainCategory),
            self.db_engine,
        )
        self.main_category_combobox = widgets.Combobox(
            placeholder="category",
            options=df.category.sort_values().tolist(),
            description="Main Category:",
            ensure_option=False,
            disabled=False,
            style=dict(description_width="initial"),
        )
        left_box_list.append(self.main_category_combobox)

        df = pd.read_sql(
            select(SubCategory),
            self.db_engine,
        )
        self.sub_category_combobox = widgets.Combobox(
            placeholder="category",
            options=df.category.sort_values().tolist(),
            description="Sub-Category:",
            ensure_option=False,
            disabled=False,
            style=dict(description_width="initial"),
        )
        left_box_list.append(self.sub_category_combobox)

        df = pd.read_sql(
            select(TimeCategory),
            self.db_engine,
        )
        self.time_category_combobox = widgets.Combobox(
            placeholder="category",
            options=df.category.sort_values().tolist(),
            description="Time-Category:",
            ensure_option=False,
            disabled=False,
            style=dict(description_width="initial"),
        )
        left_box_list.append(self.time_category_combobox)

        df = pd.read_sql(
            select(Tag),
            self.db_engine,
        )
        self.tag_multiselect = widgets.TagsInput(
            allowed_tags=df.tag.sort_values().tolist(),
            description="Available-tags:",
            allow_duplicates=False,
            disabled=False,
            rows=5,
            style=dict(description_width="initial"),
        )
        left_box_list.append(self.tag_multiselect)

        self.tags_field = widgets.Text(
            value="",
            placeholder=f"tag1{AccountingDBManager.tags_sep}tag2{AccountingDBManager.tags_sep}...",
            description="New Tags:",
            disabled=False,
            style=dict(description_width="initial"),
        )
        left_box_list.append(self.tags_field)
        left_box_list.append(
            widgets.HTML(
                value="<hr>",
            )
        )

        self.date_picker = widgets.DatePicker(
            description="Cashflow Date",
            disabled=False,
            style=dict(description_width="initial"),
        )
        left_box_list.append(self.date_picker)

        self.amount_text = widgets.FloatText(
            value=0,
            step=0.01,
            description="Cashflow Amount:",
            disabled=False,
            style=dict(description_width="initial"),
        )
        left_box_list.append(self.amount_text)

        self.description_text = widgets.Text(
            value="",
            description="Description:",
            disabled=False,
            style=dict(description_width="initial"),
        )
        left_box_list.append(self.description_text)

        add_row_button = widgets.Button(
            description="Add Cashflow",
            disabled=False,
            button_style="",
            tooltip="",
            icon="check",
            layout=button_layout,
        )
        left_box_list.append(add_row_button)
        left_box_list.append(
            widgets.HTML(
                value="<hr>",
            )
        )

        self.min_date_picker = widgets.DatePicker(
            description="Min Date",
            disabled=False,
            style=dict(description_width="initial"),
        )
        left_box_list.append(self.min_date_picker)
        self.max_date_picker = widgets.DatePicker(
            description="Max Date",
            disabled=False,
            style=dict(description_width="initial"),
        )
        left_box_list.append(self.max_date_picker)

        self.sample_frq_dropdown = widgets.Dropdown(
            options=[(eee.name, eee) for eee in SampleFrequency],
            value=SampleFrequency.Daily,
            description="Sample Frequency:",
            style=dict(description_width="initial"),
        )
        left_box_list.append(self.sample_frq_dropdown)

        show_db_button = widgets.Button(
            description="Show DB",
            disabled=False,
            button_style="",
            tooltip="",
            icon="check",
            layout=button_layout,
        )
        left_box_list.append(show_db_button)
        left_box_list.append(
            widgets.HTML(
                value="<hr>",
            )
        )

        self.select_db_entries = widgets.SelectMultiple(
            options=[],
            value=[],
            description="Select Db Entries",
            disabled=False,
            style=dict(description_width="initial"),
            layout=widgets.Layout(width="auto", height="auto"),
        )
        left_box_list.append(self.select_db_entries)
        self.tags_db_entries = widgets.TagsInput(
            allowed_tags=[],
            value=[],
            allow_duplicates=False,
            disabled=False,
            style=dict(description_width="initial"),
            layout=widgets.Layout(width="auto", height="auto"),
        )
        left_box_list.append(self.tags_db_entries)

        self.overwrite_entry_checkbox = widgets.Checkbox(
            value=True,
            description="Overwrite Entry Values",
            disabled=False,
            indent=False,
        )
        left_box_list.append(self.overwrite_entry_checkbox)
        self.delete_entry_checkbox = widgets.Checkbox(
            value=False, description="Delete Entry Values", disabled=False, indent=False
        )
        left_box_list.append(self.delete_entry_checkbox)

        update_entries_button = widgets.Button(
            description="Update DB Entries",
            disabled=False,
            button_style="",
            tooltip="",
            icon="check",
            layout=button_layout,
        )
        left_box_list.append(update_entries_button)
        left_box_list.append(
            widgets.HTML(
                value="<hr>",
            )
        )

        self.days_rolling_window_int = widgets.IntText(
            value=30,
            description="Days rolling window:",
            disabled=False,
            style=dict(description_width="initial"),
        )
        left_box_list.append(self.days_rolling_window_int)
        self.normalise_rolling_results_checkbox = widgets.Checkbox(
            value=False, description="Normalise to days", disabled=False, indent=False
        )
        left_box_list.append(self.normalise_rolling_results_checkbox)
        left_box_list.append(
            widgets.HTML(
                value="<hr>",
            )
        )

        create_backupt_button = widgets.Button(
            description="Create DB backup",
            disabled=False,
            button_style="",
            tooltip="",
            icon="check",
            layout=button_layout,
        )
        left_box_list.append(create_backupt_button)
        # endregion
        # region Right Box
        self.output_window = widgets.Output(
            layout=widgets.Layout(
                width="800px",
                overflow_x="scroll",
                height="400px",
                overflow_y="scroll",
            )
        )
        right_box_list.append(self.output_window)
        # endregion

        left_box = widgets.VBox(left_box_list)
        right_box = widgets.VBox(right_box_list)

        self.draw_output_window = widgets.Output(
            layout=widgets.Layout(
                overflow_x="scroll",
                height="400px",
                overflow_y="scroll",
            )
        )

        self.main_element = widgets.VBox(
            [widgets.HBox([left_box, right_box]), self.draw_output_window]
        )

        show_db_button.on_click(self.show_db)
        add_row_button.on_click(self.add_row)
        clear_fields_button.on_click(self.clear_fields)
        update_entries_button.on_click(self.update_entries)
        create_backupt_button.on_click(self.create_backupt)

    def display(self):
        super().display()
        self.show_db(None)

    def get_filtered_df(self):
        df = self.acc_table.get_df()
        return self.filter_df(df)

    @staticmethod
    def match_tags(input_tags, target_tags):
        if isinstance(input_tags, str):
            input_tags = input_tags.split(AccountingDBManager.tags_sep)

        for iii in input_tags:
            if iii in target_tags:
                return True
        return False

    def are_input_tags(self):
        return len(self.tags_field.value) > 0 or len(self.tag_multiselect.value) > 0

    def get_input_tags_list(self, extend_tag_options=False):
        tag_list = []
        if len(self.tags_field.value) > 0:
            tag_list = self.tags_field.value.split(AccountingDBManager.tags_sep)
            if len(tag_list) > 0 and extend_tag_options:
                self.tag_multiselect.allowed_tags = sorted(
                    list(set(self.tag_multiselect.allowed_tags + tag_list))
                )
        tag_list.extend(self.tag_multiselect.value)
        return tag_list

    def filter_df(self, input_df):
        if self.min_date_picker.value:
            min_date = self.min_date_picker.value
        else:
            min_date = date.min
        if self.max_date_picker.value:
            max_date = self.max_date_picker.value
        else:
            max_date = date.max
        past_df = input_df[(input_df.date < min_date)]
        df = input_df[(input_df.date <= max_date) & (input_df.date >= min_date)]

        if self.main_category_combobox.value:
            df = df[df.main_category == self.main_category_combobox.value]
        if self.sub_category_combobox.value:
            df = df[df.sub_category == self.sub_category_combobox.value]
        if self.time_category_combobox.value:
            df = df[df.time_category == self.time_category_combobox.value]
        if self.are_input_tags():
            df = df[
                df.tag.apply(
                    lambda x: AccountingDBManager.match_tags(
                        x, self.get_input_tags_list()
                    )
                )
            ]

        if self.description_text.value:
            df = df[
                df.description.apply(lambda x: x.lower()).str.contains(
                    self.description_text.value
                )
            ]

        return df, past_df, min_date

    @staticmethod
    def go_to_end_of(input_date, frq):
        current_date = input_date
        if frq == SampleFrequency.Weekly:
            return input_date + Week(weekday=6)
        elif frq == SampleFrequency.Monthly:
            return input_date + MonthEnd(0)
        elif frq == SampleFrequency.Yearly:
            while current_date.year == input_date.year:
                current_date = input_date
                input_date += timedelta(days=1)
        elif frq == SampleFrequency.Daily:
            pass
        elif frq == SampleFrequency.Quarterly:
            while (current_date.month - 1) // 3 == (input_date.month - 1) // 3:
                current_date = input_date
                input_date += timedelta(days=1)
        else:
            raise Exception("Unrecognised sample frequency")

        return current_date

    @staticmethod
    def diff_year(d1, d2):
        tot_days = (d1 - d2).days
        return tot_days / 365

    @staticmethod
    def diff_month(d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month + 1

    @staticmethod
    def diff_week(d2, d1):
        monday1 = d1 - timedelta(days=d1.weekday())
        monday2 = d2 - timedelta(days=d2.weekday())

        return (monday2 - monday1).days / 7 + 1

    @staticmethod
    def create_rolling_plot(input_df, days_offset=30, normalise=False):
        ofs = f"{days_offset}D"
        fil_df = input_df[input_df.sub_category != "Initial"]

        all_ser = []
        all_ser.append(
            fil_df[fil_df.main_category == "Income"]
            .groupby("date")
            .apply(lambda x: x.amount.sum())
            .rename(f"income-{ofs}")
        )
        all_ser.append(
            fil_df.groupby("date").apply(lambda x: x.amount.sum()).rename(f"net-{ofs}")
        )
        all_ser.append(
            fil_df[fil_df.main_category == "Outcome"]
            .groupby("date")
            .apply(lambda x: -x.amount.sum())
            .rename(f"outcome-{ofs}")
        )

        box_fig = go.Figure()
        violin_fig = go.Figure()
        fig = go.Figure()
        min_y = 0
        max_y = 0
        for ser in all_ser:
            ser.sort_index(inplace=True)
            ser.index = pd.to_datetime(ser.index)

            ofs_int = int(ofs.replace("D", ""))
            s1 = ser.rolling(window=ofs).sum()

            start_date = s1.index.min() + pd.Timedelta(ofs_int, unit="D")
            s1 = s1[s1.index > start_date]
            if normalise:
                s1 = s1 / ofs_int

            fig.add_trace(
                go.Scatter(
                    x=s1.index,
                    y=s1.values,
                    mode="lines",
                    hovertemplate="%{y:,.2f}",
                    name=ser.name,
                )
            )
            mu = s1.mean()
            sigma = s1.std()
            fact = 2
            min_y = min(min_y, mu - fact * sigma)
            max_y = max(max_y, mu + fact * sigma)
            print(
                f"{ser.name:<15}: {mu:,.2f}"
                + " \u00B1 "
                + f"{sigma:,.2f}; q1%->{s1.quantile(0.01):,.2f}, q10%->{s1.quantile(0.1):,.2f}, q50%->{s1.quantile(0.5):,.2f}, q90%->{s1.quantile(0.9):,.2f}, q99%->{s1.quantile(0.99):,.2f}"
            )

            box_fig.add_trace(go.Box(y=s1.values, name=ser.name, boxmean="sd"))

            violin_fig.add_trace(
                go.Violin(
                    y=s1.values, name=ser.name, box_visible=False, meanline_visible=True
                )
            )

        fig.update_layout(hovermode="x", yaxis_range=[0, max_y])
        box_fig.update_layout(
            # autosize=False,
            width=700,
            height=800,
        )
        violin_fig.update_layout(
            # autosize=False,
            width=700,
            height=800,
        )
        return fig, box_fig, violin_fig

    @staticmethod
    def get_income_analysis(input_df):
        if len(input_df.index) == 0:
            return 0, 0, 1, 1, 1, 1

        tmp_max_date = input_df.date.max()
        tmp_min_date = input_df.date.min()

        frac_years = AccountingDBManager.diff_year(tmp_max_date, tmp_min_date)
        tot_months = AccountingDBManager.diff_month(tmp_max_date, tmp_min_date)
        tot_weeks = AccountingDBManager.diff_week(tmp_max_date, tmp_min_date)
        tot_days = (tmp_max_date - tmp_min_date).days

        net_income = input_df[
            input_df.main_category == EnumMainCat.Income.name
        ].amount.sum()
        net_outcome = input_df[
            input_df.main_category == EnumMainCat.Outcome.name
        ].amount.sum()

        in_df = input_df[input_df.sub_category == "Initial"]
        if len(in_df.index) > 0:
            net_income -= in_df.amount.sum()

        return net_income, net_outcome, frac_years, tot_months, tot_weeks, tot_days

    # region Button actions
    def create_backupt(self, _):
        self.acc_table.create_backup()

    def clear_fields(self, _):
        self.date_picker.value = None
        self.amount_text.value = 0
        self.description_text.value = ""

        self.main_category_combobox.value = ""
        self.sub_category_combobox.value = ""
        self.time_category_combobox.value = ""
        self.tags_field.value = ""
        self.tag_multiselect.value = []
        self.select_db_entries.value = []
        self.tags_db_entries.value = []

        self.min_date_picker.value = None
        self.max_date_picker.value = None

    def update_entries(self, _):

        sel_ids = [int(eee.split("-")[0]) for eee in self.select_db_entries.value]
        sel_ids.extend([int(eee.split("-")[0]) for eee in self.tags_db_entries.value])
        sel_ids = sorted(list(set(sel_ids)))
        with Session(self.db_engine) as session:
            if self.delete_entry_checkbox.value:
                ddd = delete(Cashflow).where(Cashflow.id.in_(sel_ids))
                session.execute(ddd)
            else:
                sss = select(Cashflow).where(Cashflow.id.in_(sel_ids))
                for ccc in session.execute(sss).scalars():
                    if self.amount_text.value:
                        ccc.amount = self.amount_text.value
                    if self.date_picker.value:
                        ccc.date = self.date_picker.value
                    if self.description_text.value:
                        ccc.description = self.description_text.value

                    if self.main_category_combobox.value:
                        ccc.main_category = Cashflow.project_element(
                            session, self.main_category_combobox.value, "main_category"
                        )
                    if self.sub_category_combobox.value:
                        ccc.sub_category = Cashflow.project_element(
                            session, self.sub_category_combobox.value, "sub_category"
                        )
                    if self.time_category_combobox.value:
                        ccc.time_category = Cashflow.project_element(
                            session, self.time_category_combobox.value, "time_category"
                        )
                    if self.are_input_tags():
                        tag_list = Cashflow.project_element(
                            session, self.get_input_tags_list(), "tags"
                        )
                        if self.overwrite_entry_checkbox.value:
                            ccc.tags = tag_list
                        else:
                            tag_list.extend(ccc.tags)
                            ccc.tags = list(set(tag_list))
            session.commit()
        self.show_db(None)

    def add_row(self, _):
        with Session(self.db_engine) as session:
            cfl_entry = Cashflow(
                session=session,
                date=self.date_picker.value,
                amount=self.amount_text.value,
                description=self.description_text.value,
                main_category=self.main_category_combobox.value,
                sub_category=self.sub_category_combobox.value,
                time_category=self.time_category_combobox.value,
                tags=self.get_input_tags_list(extend_tag_options=True),
            )
            session.add(cfl_entry)
            session.commit()
        self.show_db(None)

    def get_fit_df(self, input_df, past_amnt):
        fil_df = pd.get_dummies(
            input_df, columns=["main_category", "sub_category", "time_category"]
        )

        sampl_frq = SampleFrequency.Weekly
        if sampl_frq == SampleFrequency.Monthly:
            d_frq = pd.offsets.MonthEnd()
            num_bus_days = 12
            ido1 = [1 if (iii in [0, 1, 2]) else 0 for iii in range(53)]
            ido2 = [1 if (iii in [0, 1, 2]) else 0 for iii in range(53)]
        elif sampl_frq == SampleFrequency.Weekly:
            d_frq = pd.offsets.Week(weekday=6)
            num_bus_days = 52
            ido1 = [1 if (iii in [0, 1, 2, 3]) else 0 for iii in range(53)]
            ido2 = [1 if (iii in [0, 1, 2, 3]) else 0 for iii in range(53)]
        else:
            raise Exception()

        fil_df.date = fil_df.date.apply(
            lambda x: AccountingDBManager.go_to_end_of(x, sampl_frq)
        )

        grp_df = fil_df.groupby("date")

        all_df = []
        all_df.append(grp_df.apply(lambda x: x.amount.sum()).rename("cashflow"))

        cols = list(fil_df.columns)
        cols.remove("id")
        cols.remove("date")
        cols.remove("amount")
        cols.remove("description")
        if "tag" in cols:
            cols.remove("tag")

        for ccc in cols:
            all_df.append(grp_df.apply(lambda x: x[ccc].sum()).rename(ccc))

        f_df = pd.concat(all_df, axis=1)

        f_df.index = pd.to_datetime(f_df.index)
        f_df["cashflow"] = f_df.cashflow.cumsum() + past_amnt

        fit_df = f_df.reindex(
            pd.date_range(start=f_df.index.min(), end=f_df.index.max(), freq=d_frq),
            fill_value=0,
        )

        # Variables
        endog = fit_df.loc[:, "cashflow"]

        exodg_list = list(fit_df.columns)
        exodg_list.remove("cashflow")
        exog = sm.add_constant(fit_df.loc[:, exodg_list])

        # Fit the model
        mod = sm.tsa.statespace.SARIMAX(
            endog,
            exog=exog,
            order=(ido1, 1, ido2),
        )
        fit_res = mod.fit(disp=False, maxiter=200, method="powell")

        predict = fit_res.get_prediction()
        predict_ci = predict.conf_int(alpha=0.05)
        predict_ci["predicted_mean"] = predict.predicted_mean

        poss_regr_frq = fit_df.drop("cashflow", axis=1)
        poss_regr_frq = poss_regr_frq.value_counts(normalize=True)

        num_sim = 1000

        all_ser = []
        for iii in range(num_sim):
            ex1 = pd.DataFrame.from_records(
                np.random.choice(
                    poss_regr_frq.index.values, num_bus_days, p=poss_regr_frq.values
                ),
                columns=poss_regr_frq.index.names,
            )
            ex1 = sm.add_constant(ex1, has_constant="add")

            fore = fit_res.get_forecast(steps=len(ex1.index), exog=ex1)
            fore_ci = fore.conf_int(alpha=0.05)
            fore_ci["predicted_mean"] = fore.predicted_mean

            all_ser.append(fore_ci)

        pred_list = []
        pred_list.append("lower cashflow")
        pred_list.append("predicted_mean")
        pred_list.append("upper cashflow")
        e_list = []
        for nnn in pred_list:
            tmp_1 = []
            for ddd in all_ser:
                tmp_1.append(ddd[nnn])
            e1 = pd.concat(tmp_1, axis=1)
            e_list.append(e1.mean(axis=1).rename(nnn))
        fore_ci = pd.concat(e_list, axis=1)

        # regr_df = pd.concat([fore_ci, predict_ci])
        # regr_df.sort_index(inplace=True)

        fore_ci = pd.concat([fore_ci, predict_ci.iloc[-1].to_frame().T])
        fore_ci.sort_index(inplace=True)

        return predict_ci, fore_ci

    def show_db(self, _):
        df = self.acc_table.get_df()
        df, past_df, min_date = self.filter_df(df)
        base_amnt = past_df.amount.sum()

        (
            net_income,
            net_outcome,
            frac_years,
            tot_months,
            tot_weeks,
            tot_days,
        ) = AccountingDBManager.get_income_analysis(df)

        norm_df = df[
            (df.time_category != EnumTimeCat.Exceptional.name)
            & (df.time_category != EnumTimeCat.OneTime.name)
        ]
        (
            norm_net_income,
            norm_net_outcome,
            _,
            _,
            _,
            _,
        ) = AccountingDBManager.get_income_analysis(norm_df)

        self.output_window.clear_output()
        with self.output_window:
            styled_df = (
                df.style.applymap(
                    lambda x: "font-weight: bold; color: green"
                    if x > 0
                    else "font-weight: bold; color: red",
                    subset=pd.IndexSlice[:, ["amount"]],
                )
                .format("{:,.2f}", na_rep="MISS", subset=pd.IndexSlice[:, ["amount"]])
                .apply(
                    lambda x: [
                        "background: #b3ffb3;"
                        if x["main_category"] == EnumMainCat.Income.name
                        else "background: #ffb3b3;"
                    ]
                    * len(x.index),
                    axis=1,
                )
            )
            cell_hover = {  # for row hover use <tr> instead of <td>
                "selector": "td:hover",
                "props": [("background-color", "#ffffb3")],
            }
            styled_df.set_table_styles([cell_hover])

            ttips = df.apply(
                lambda x: pd.Series([x.tag] * len(x.index), index=df.columns),
                axis=1,
            )
            styled_df.set_tooltips(ttips)
            df.drop(["tag"], inplace=True, axis=1)

            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                display(styled_df)

        df.sort_values("id", inplace=True, axis=0, ignore_index=True)
        self.select_db_entries.options = df.apply(
            lambda x: f"{x.id}-{x.description[:7]}-{x.main_category}-{x.sub_category}",
            axis=1,
        )
        self.tags_db_entries.allowed_tags = self.select_db_entries.options

        self.draw_output_window.clear_output()
        with self.draw_output_window:
            tmp_init_df = df[df.sub_category == EnumSubCat.Initial.name]
            if len(tmp_init_df.index) > 0:
                base_amnt += tmp_init_df.amount.sum()
                min_date = tmp_init_df.date.min()
            df = df[~(df.sub_category == EnumSubCat.Initial.name)]

            regr_df, forecast_df = self.get_fit_df(df, base_amnt)

            df.date = df.date.apply(
                lambda x: AccountingDBManager.go_to_end_of(
                    x, self.sample_frq_dropdown.value
                )
            )
            ser = (
                df.groupby("date")
                .apply(lambda x: x.amount.sum())
                .rename("bank account")
            )
            ser.sort_index(inplace=True)
            ser_cum_sum = ser.cumsum()

            print(f"Base on {min_date:%d-%m-%Y}: {base_amnt:,.2f}")
            print(
                f"Saldo on {ser_cum_sum.index[-1]:%d-%m-%Y}: {ser_cum_sum[-1]:,.2f} ({ser_cum_sum[-1]+base_amnt:,.2f})"
            )
            dict_1 = {}
            dict_1["Yearly"] = {"value": frac_years, "fmt": ".2f"}
            dict_1["Monthly"] = {"value": tot_months, "fmt": "d"}
            dict_1["Weekly"] = {"value": tot_weeks, "fmt": ".2f"}
            dict_1["Daily"] = {"value": tot_days, "fmt": "d"}
            for kkk, vvv in dict_1.items():
                tmp_s = f'{kkk:<7}({vvv["value"]:{vvv["fmt"]}})'
                print(
                    f'{tmp_s:<15} Income/Outcome/Net: {net_income/vvv["value"]:,.2f}, {net_outcome/vvv["value"]:,.2f}, {(net_income+net_outcome)/vvv["value"]:,.2f}'
                )

            print("--- Just with normal cashflows ---")
            print(
                f"Saldo on {norm_df.date.max():%d-%m-%Y}: {norm_df.amount.sum():,.2f} ({norm_df.amount.sum()+base_amnt:,.2f})"
            )
            for kkk, vvv in dict_1.items():
                tmp_s = f'{kkk:<7}({vvv["value"]:{vvv["fmt"]}})'
                print(
                    f'{tmp_s:<15} Income/Outcome/Net: {norm_net_income/vvv["value"]:,.2f}, {norm_net_outcome/vvv["value"]:,.2f}, {(norm_net_income+norm_net_outcome)/vvv["value"]:,.2f}'
                )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=ser_cum_sum.index,
                    y=ser_cum_sum.values + base_amnt,
                    line=dict(color="blue"),
                    hovertemplate="%{y:,.2f} €",
                    name="bank account",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=regr_df["predicted_mean"].index,
                    y=regr_df["predicted_mean"].values,
                    legendgroup="regression",
                    hovertemplate="%{y:,.2f} €",
                    line=dict(color="green"),
                    name="regr mean",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=regr_df["upper cashflow"].index,
                    y=regr_df["upper cashflow"].values,
                    line=dict(color="green"),
                    hovertemplate="%{y:,.2f} €",
                    legendgroup="regression",
                    name="95% quantile",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=regr_df["lower cashflow"].index,
                    y=regr_df["lower cashflow"].values,
                    line=dict(color="green"),
                    hovertemplate="%{y:,.2f} €",
                    legendgroup="regression",
                    fill="tonexty",
                    name="5% quantile",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast_df["predicted_mean"].index,
                    y=forecast_df["predicted_mean"].values,
                    legendgroup="forecast",
                    hovertemplate="%{y:,.2f} €",
                    line=dict(color="red"),
                    name="forecast mean",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_df["upper cashflow"].index,
                    y=forecast_df["upper cashflow"].values,
                    line=dict(color="red"),
                    hovertemplate="%{y:,.2f} €",
                    legendgroup="forecast",
                    name="95% quantile",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_df["lower cashflow"].index,
                    y=forecast_df["lower cashflow"].values,
                    line=dict(color="red"),
                    hovertemplate="%{y:,.2f} €",
                    legendgroup="forecast",
                    fill="tonexty",
                    name="5% quantile",
                )
            )

            fig.update_layout(
                hovermode="x",
                height=700,
            )

            fig.show()

            ### Waterfall
            years = []
            quarters = []
            months = []
            weeks = []
            days = []
            measures = []
            values = []
            for ido, vvv in ser.iteritems():
                if vvv != 0:
                    years.append(ido.year)
                    quarters.append(f"q{(ido.month-1)//3 + 1} - {ido.year}")
                    months.append(f'{ido.strftime("%b")} - {ido.year}')
                    weeks.append(f"{ido.isocalendar().week} - {ido.year}")
                    days.append(f"{ido.timetuple().tm_yday} - {ido.year}")

                    measures.append("relative")
                    values.append(vvv)

            sel_tune = None
            if self.sample_frq_dropdown.value == SampleFrequency.Daily:
                sel_tune = days
            elif self.sample_frq_dropdown.value == SampleFrequency.Weekly:
                sel_tune = weeks
            elif self.sample_frq_dropdown.value == SampleFrequency.Monthly:
                sel_tune = months
            elif self.sample_frq_dropdown.value == SampleFrequency.Yearly:
                sel_tune = years
            elif self.sample_frq_dropdown.value == SampleFrequency.Quarterly:
                sel_tune = quarters
            else:
                raise Exception("Unrecognised sample frequency")

            fig = go.Figure(
                go.Waterfall(
                    x=[years, sel_tune],
                    measure=measures,
                    y=values,
                    base=base_amnt,
                )
            )

            fig.update_layout(waterfallgap=0.3)
            fig.show()

            ### SunBurst
            df = self.acc_table.get_df(
                aggregate_tags=False, normalised_by_tags=True, add_amount_sign=False
            )
            df, past_df, min_date = self.filter_df(df)

            fig = px.sunburst(
                df,
                path=["main_category", "sub_category", "time_category", "tag"],
                values="amount",
            )
            fig.show()

            df, past_df, min_date = self.get_filtered_df()
            fig, box_fig, violin_fig = self.create_rolling_plot(
                df,
                days_offset=self.days_rolling_window_int.value,
                normalise=self.normalise_rolling_results_checkbox.value,
            )
            box_fig.show()
            violin_fig.show()
            fig.show()

    # endregion
