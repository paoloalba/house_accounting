import os
import sys
import json
import dash

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import plotly.express as px

from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.sql import select, delete

sys.path.append(os.getenv("GLOBAL_LIBRARIES_PATH", ""))
sys.path = list(set(sys.path))
from house_accounting.db_handlers import AccountingTable
from house_accounting.enumerators import MainCategory, SubCategory, TimeCategory
from house_accounting.enumerators import SampleFrequency
from house_accounting.widgets import AccountingDBManager
from config import global_config
from house_accounting.models import Cashflow

try:
    from generic_helpers.row_filters import filters
    from generic_helpers.generic import parse_extracted_csv
except:
    filters = {}
    def parse_extracted_csv(*args, **kwargs):
        raise NotImplementedError()

original_pmt_storage = os.getenv("PMT_STG_PATH")
pmt_storage = os.path.join(original_pmt_storage, "house_accounting")
os.makedirs(pmt_storage, exist_ok=True)

db_path = os.path.join(pmt_storage, global_config["example_db_name"])
acc_table = AccountingTable(db_path)

### Helpers
def create_table_col_spec(input_df):
    datatable_cols = []
    for i in input_df.columns:
        tmp_dict = {"name": i, "id": i}
        if i == "amount":
            tmp_dict["type"] = "numeric"
            tmp_dict["format"] = dict(specifier=",.2f")
            tmp_dict["filter_options"] = dict(case="sensitive")
        elif i == "date":
            tmp_dict["type"] = "datetime"
        elif i in ["main_category", "sub_category", "time_category"]:
            tmp_dict["presentation"] = "dropdown"
            if i == "main_category":
                tmp_dict["editable"] = False
        elif i == "id":
            tmp_dict["editable"] = False
            tmp_dict["type"] = "numeric"
            tmp_dict["filter_options"] = dict(case="sensitive")
        datatable_cols.append(tmp_dict)
    return datatable_cols


def get_fit_df(input_df, past_amnt, sampl_frq):
    fil_df = pd.get_dummies(
        input_df, columns=["main_category", "sub_category", "time_category"]
    )

    if sampl_frq == SampleFrequency.Monthly:
        d_frq = pd.offsets.MonthEnd()
        num_bus_days = 12
        ido1 = [1 if (iii in [0, 1, 2]) else 0 for iii in range(num_bus_days + 1)]
        ido2 = [1 if (iii in [0, 1, 2]) else 0 for iii in range(num_bus_days + 1)]
    elif sampl_frq == SampleFrequency.Weekly:
        d_frq = pd.offsets.Week(weekday=6)
        num_bus_days = 52
        ido1 = [1 if (iii in [0, 1, 2, 3]) else 0 for iii in range(num_bus_days + 1)]
        ido2 = [1 if (iii in [0, 1, 2, 3]) else 0 for iii in range(num_bus_days + 1)]
    else:
        raise Exception(f"Unsupported frequency {sampl_frq}")

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


def diff_dashtable(new_data, old_df):

    new_df = pd.DataFrame(data=new_data)

    new_df["date"] = pd.to_datetime(new_df["date"])
    new_df["amount"] = new_df["amount"].apply(
        lambda x: np.round(x, 2) if isinstance(x, float) else x
    )

    old_df["date"] = pd.to_datetime(old_df["date"])
    old_df["amount"] = old_df["amount"].apply(
        lambda x: np.round(x, 2) if isinstance(x, float) else x
    )

    return (
        new_df[~new_df.apply(tuple, 1).isin(old_df.apply(tuple, 1))],
        old_df[~old_df.apply(tuple, 1).isin(new_df.apply(tuple, 1))],
    )


def filter_rows(inp_obj):
    for _, fff in filters.items():
        try:
            return pd.Series(fff.map_row(inp_obj))
        except:
            pass
    raise Exception(f"Impossible to filter row:\n{inp_obj}")


def get_src_matching(input_sss, old_df):
    mtch = old_df[
        (np.round(old_df["amount"], 2) == np.round(input_sss["amount"], 2))
        & (old_df["date"] == input_sss["date"])
    ]
    return len(mtch.index)


#######

app = Dash(__name__)

### Main Table
dropdown = {}
dropdown["main_category"] = dict(
    options=[{"label": i.name, "value": i.name} for i in list(MainCategory)]
)
dropdown["sub_category"] = dict(
    options=[{"label": i.name, "value": i.name} for i in list(SubCategory)]
)
dropdown["time_category"] = dict(
    options=[{"label": i.name, "value": i.name} for i in list(TimeCategory)]
)

base_sort_list = [{"column_id": "date", "direction": "desc"}]

tmp_df = acc_table.get_df()
tmp_df["date"] = pd.to_datetime(tmp_df["date"])
main_table = dash_table.DataTable(
    columns=create_table_col_spec(tmp_df),
    data=tmp_df.to_dict("records"),
    editable=True,
    row_deletable=True,
    filter_action="native",
    sort_action="native",
    sort_mode="multi",
    # row_selectable="multi",
    export_format="xlsx",
    page_action="native",
    page_current=0,
    page_size=10,
    dropdown=dropdown,
    filter_options=dict(case="insensitive"),
    sort_by=base_sort_list,
    style_data_conditional=[
        {
            'if': {
                'filter_query': '{time_category} contains "Exceptional"',
                'column_id': 'time_category',
            },
            'backgroundColor': '#ff8080',
            'color': 'white',
        },
        {
            'if': {
                'filter_query': '{time_category} contains "Recurring"',
                'column_id': 'time_category',
            },
            'backgroundColor': '#85e085',
            'color': 'white',
        },
        {
            'if': {
                'filter_query': '{time_category} contains "OneTime"',
                'column_id': 'time_category',
            },
            'backgroundColor': '#ffff66',
            'color': 'black',
        },
        {
            'if': {
                'filter_query': '{main_category} contains "Outcome"',
                'column_id': ['amount', 'main_category'],
            },
            'backgroundColor': '#ff3333',
            'color': 'white',
        },
        {
            'if': {
                'filter_query': '{main_category} contains "Income"',
                'column_id': ['amount', 'main_category'],
            },
            'backgroundColor': '#70db70',
            'color': 'white',
        },
    ],
)

custom_filter_radio = dcc.RadioItems(
    [{'label': 'Read filter_query', 'value': 'read'}, {'label': 'Write to filter_query', 'value': 'write'}],
    'read',
    id='custom_filter_radio',
)
filter_query_input = dcc.Input(id='filter_query_input', placeholder='Enter filter query')
filter_query_output = html.Div(id='filter_query_output')


### graphs
plot_container = html.Div()
regression_series_store = dcc.Store(id="regression_series_store")
forecast_series_store = dcc.Store(id="forecast_series_store")
main_df_store = dcc.Store(id="main_df_store")
base_amnt_store = dcc.Store(id="base_amnt_store")
plot_aggr_dropdown = dcc.Dropdown(
    options=[
        {"label": SampleFrequency.Daily.name, "value": SampleFrequency.Daily.name},
        {"label": SampleFrequency.Weekly.name, "value": SampleFrequency.Weekly.name},
        {"label": SampleFrequency.Monthly.name, "value": SampleFrequency.Monthly.name},
        {
            "label": SampleFrequency.Quarterly.name,
            "value": SampleFrequency.Quarterly.name,
        },
        {"label": SampleFrequency.Yearly.name, "value": SampleFrequency.Yearly.name},
    ],
    value=SampleFrequency.Daily.name,
    id="plot_aggr_dropdown",
    clearable=False,
    style=dict(
                    width='40%',
                    display='inline-block',
                    verticalAlign="middle"
                )
)
plot_moving_avg_container = html.Div()
mvg_avg_days_offset_input = dcc.Input(
    id="mvg_avg_days_offset_input",
    type="number",
    value=30,
    min=0,
    step=1,
    max=None,
)
regr_aggr_dropdown = dcc.Dropdown(
    options=[
        {"label": SampleFrequency.Daily.name, "value": SampleFrequency.Daily.name},
        {"label": SampleFrequency.Weekly.name, "value": SampleFrequency.Weekly.name},
        {"label": SampleFrequency.Monthly.name, "value": SampleFrequency.Monthly.name},
        {
            "label": SampleFrequency.Quarterly.name,
            "value": SampleFrequency.Quarterly.name,
        },
        {"label": SampleFrequency.Yearly.name, "value": SampleFrequency.Yearly.name},
    ],
    value=SampleFrequency.Weekly.name,
    id="regr_aggr_dropdown",
    clearable=False,
    style=dict(
                    width='40%',
                    display='inline-block',
                    verticalAlign="middle"
                )
)

### buttons
add_row_button = html.Button("Add Row", id="add_row_button")
update_regression_button = html.Button(
    "Update Regression", id="update_regression_button"
)
show_diff_table_button = html.Button("Diff DataTable", id="show_diff_table_button")
update_database_button = html.Button("Update Database", id="update_database_button")
reset_filters_button = html.Button("Reset filters", id="reset_filters_button")
refresh_main_df_button = html.Button("Refresh main df", id="refresh_main_df_button")
create_db_backup_button = html.Button("Create backup", id="create_db_backup_button")

### Text
summary_text = dcc.Markdown(style={"white-space": "pre"})
forecast_text = dcc.Markdown(style={"white-space": "pre"})
data_diff = html.Div(id="data_diff")
db_info_text = html.Div(id="db_info_text")
mvg_avg_info_text = html.Div()

### stores
cached_df_store = dcc.Store(id="cached_df_store", data=tmp_df.to_json())
data_diff_store = dcc.Store(id="data_diff_store")
removed_data_diff_store = dcc.Store(id="removed_data_diff_store")

### upload
upload_csv = dcc.Upload(
    id="upload_csv",
    children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
    style={
        "width": "100%",
        "height": "60px",
        "lineHeight": "60px",
        "borderWidth": "1px",
        "borderStyle": "dashed",
        "borderRadius": "5px",
        "textAlign": "center",
        "margin": "10px",
    },
    multiple=True,
)

### main layouy

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        refresh_main_df_button,
                        add_row_button,
                        show_diff_table_button,
                        update_database_button,
                        reset_filters_button,
                        create_db_backup_button,
                    ]
                ),
                upload_csv,
                html.Br(),
                db_info_text,
                html.Div([
                    custom_filter_radio,
                    filter_query_input,
                    filter_query_output,
                ]),
                main_table,
                html.Hr(),
                data_diff,
                html.Hr(),
                summary_text,
                forecast_text,
                html.Hr(),
                html.Div(
                    [
                        update_regression_button,
                        html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H4("""Data aggregation frequency""",
                                                    style={'margin-right': '2em'})
                                        ],
                                    ),
                                    plot_aggr_dropdown,
                                ],
                                style=dict(display='flex')
                        ),
                        html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H4("""Regression basis aggregation frequency""",
                                                    style={'margin-right': '2em'})
                                        ],
                                    ),
                                    regr_aggr_dropdown,
                                ],
                                style=dict(display='flex')
                        ),
                    ]
                ),
                plot_container,
                mvg_avg_days_offset_input,
                mvg_avg_info_text,
                plot_moving_avg_container,
                regression_series_store,
                forecast_series_store,
                main_df_store,
                base_amnt_store,
                data_diff_store,
                removed_data_diff_store,
                cached_df_store,
            ]
        )
    ]
)

### Callbacks

@app.callback(
    Output(filter_query_input, 'style'),
    Output(filter_query_output, 'style'),
    Input(custom_filter_radio, 'value')
)
def query_input_output(val):
    input_style = {'width': '100%'}
    output_style = {}
    if val == 'read':
        input_style.update(display='none')
        output_style.update(display='inline-block')
    else:
        input_style.update(display='inline-block')
        output_style.update(display='none')
    return input_style, output_style

@app.callback(
    Output(filter_query_output, 'children'),
    Input(main_table, 'filter_query')
)
def read_query(query):
    if query is None:
        return "No filter query"
    return dcc.Markdown('`filter_query = "{}"`'.format(query))

@app.callback(
    Output(db_info_text, "children"),
    [Input(create_db_backup_button, "n_clicks")],
)
def create_db_backup(n_clicks):
    md_list = []
    md_list.append(dcc.Markdown(f"View on {acc_table.db_path}"))
    if n_clicks is not None:
        bckp_name = acc_table.create_backup()
        md_list.append(dcc.Markdown(f"Created db backup: {bckp_name}"))
    return md_list


@app.callback(
    Output(main_table, "filter_query"),
    Output(main_table, "sort_by"),
    [
        Input(reset_filters_button, "n_clicks"),
        Input(filter_query_input, 'value'),
    ],
    [
        State(main_table, "filter_query"),
        State(main_table, "sort_by"),
    ],
)
def update_filters(reset_filters_n_clicks, input_filter_query, current_filter_query, current_sort_by):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "reset_filters_button":
        if reset_filters_n_clicks is None:
            return "" if current_filter_query is None else current_filter_query, current_sort_by
        return "", base_sort_list
    elif trigger == "filter_query_input":
        if input_filter_query is None:
            return current_filter_query, current_sort_by
        return input_filter_query, current_sort_by
    else:
        raise PreventUpdate


@app.callback(
    Output(data_diff, "children"),
    Output(data_diff_store, "data"),
    Output(removed_data_diff_store, "data"),
    [
        Input(show_diff_table_button, "n_clicks"),
        Input(update_database_button, "n_clicks"),
        Input(upload_csv, "contents"),
        Input(upload_csv, "filename"),
    ],
    [
        State(main_table, "data"),
        State(data_diff_store, "data"),
        State(removed_data_diff_store, "data"),
        State(cached_df_store, "data"),
    ],
)
def update_output(
    n_clicks_show,
    n_clicks_update,
    list_of_contents,
    list_of_filenames,
    data,
    data_diff,
    removed_data_diff,
    cached_df_store_data,
):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "update_database_button":
        if n_clicks_update is None:
            raise PreventUpdate

        dff = pd.read_json(data_diff, typ="frame")
        dff["tag"] = dff["tag"].apply(lambda x: x if isinstance(x, list) else x.split(";"))
        dff.rename(columns={"tag": "tags"}, inplace=True)
        if removed_data_diff:
            rem_dff = pd.read_json(removed_data_diff, typ="frame")
        else:
            rem_dff = None

        with Session(acc_table.db_engine) as session:
            new_entries = 0
            updated_entries = 0
            removed_entries = 0
            for ido, row in dff.iterrows():
                if ("id" in row.index) and row["id"]:
                    sel_ids = [row["id"]]

                    sss = select(Cashflow).where(Cashflow.id.in_(sel_ids))
                    for ccc in session.execute(sss).scalars():
                        for index, value in row.items():
                            if index in Cashflow._unique_check_dict:
                                new_elem = Cashflow.project_element(
                                    session, value, index
                                )
                            else:
                                new_elem = value
                            setattr(ccc, index, new_elem)
                    updated_entries += 1
                else:
                    if row.amount > 0:
                        main_cat = MainCategory.Income
                    else:
                        main_cat = MainCategory.Outcome

                    if isinstance(row.tags, list):
                        tmp_tag = row.tags
                    else:
                        tmp_tag = row.tags.split(";")

                    if isinstance(row.description, str):
                        tmp_descr = row.description
                    else:
                        tmp_descr = ""

                    cfl_entry = Cashflow(
                        session=session,
                        date=row.date,
                        amount=row.amount,
                        description=tmp_descr,
                        main_category=main_cat,
                        sub_category=row.sub_category,
                        time_category=row.time_category,
                        tags=tmp_tag,
                    )
                    session.add(cfl_entry)
                    new_entries += 1
            if rem_dff is not None:
                for ido, row in rem_dff.iterrows():
                    if ("id" in row.index) and row["id"]:
                        sel_ids = [row["id"]]
                        sss = select(Cashflow).where(Cashflow.id.in_(sel_ids))
                        for ccc in session.execute(sss).scalars():
                            ccc.tags = []

                        ddd = delete(Cashflow).where(Cashflow.id.in_(sel_ids))
                        session.execute(ddd)
                        removed_entries += 1
            session.commit()
        return (
            f"Updated {len(dff.index)} entries: updated -> {updated_entries}, new -> {new_entries}, removed -> {removed_entries}",
            None,
            None,
        )
    elif trigger == "upload_csv":
        if list_of_contents is not None:

            res_df = parse_extracted_csv(zip(list_of_contents, list_of_filenames))

            res_df = res_df.apply(filter_rows, axis=1)

            msk = res_df.apply(
                get_src_matching,
                axis=1,
                args=(pd.read_json(cached_df_store_data, typ="frame"),),
            )
            allo = res_df[msk == 0].copy()
            allo.sub_category = allo.sub_category.apply(lambda x: x.name).astype(
                "category"
            )
            allo.time_category = allo.time_category.apply(lambda x: x.name).astype(
                "category"
            )
            allo.tags = allo.tags.apply(
                lambda x: [eee.lower().replace(" ", "") for eee in x]
            ).apply(sorted)

            allo.rename(columns={"tags": "tag"}, inplace=True)

            md_list = []
            md_list.append(dcc.Markdown(f"Found {len(allo.index)} new entries"))

            dt_df = allo.copy()
            dt_df.tag = dt_df.tag.apply(lambda x: ";".join(x))
            tmp_table = dash_table.DataTable(
                columns=create_table_col_spec(dt_df),
                data=dt_df.to_dict("records"),
                editable=False,
                row_deletable=False,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                page_action="native",
                page_current=0,
                page_size=10,
                filter_options=dict(case="insensitive"),
            )
            md_list.append(tmp_table)

            return md_list, allo.to_json(), None
        else:
            return "No new elements found wrt DataTable", None, None
    elif trigger == "show_diff_table_button":
        if n_clicks_show is None:
            raise PreventUpdate

        diff_store_data, removed_data = diff_dashtable(
            data, pd.read_json(cached_df_store_data, typ="frame")
        )

        if len(diff_store_data.index) > 0 or len(removed_data.index) > 0:
            md_list = []

            if len(removed_data.index) > 0 and len(diff_store_data.index) > 0:
                msk_2 = removed_data.id.isin(diff_store_data.id)
                pre_mod_data = removed_data[msk_2]
                removed_data = removed_data[~msk_2]

                all_df = []
                for ido, rrr in pre_mod_data.iterrows():
                    or_r = diff_store_data[diff_store_data["id"] == rrr["id"]]
                    tmp_dict = {}
                    for iii, vvv in or_r.items():
                        v1 = vvv.values[0]
                        v2 = rrr[iii]
                        if v1 == v2:
                            tmp_dict[iii] = str(v1)
                        else:
                            tmp_dict[iii] = f"{v2} -> {v1}"
                    all_df.append(tmp_dict)
                pre_mod_data = pd.DataFrame.from_records(all_df)
            else:
                pre_mod_data = diff_store_data.copy()

            if len(pre_mod_data.index) > 0:
                dt_df = pre_mod_data.copy()
                tmp_table = dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in dt_df.columns],
                    data=dt_df.to_dict("records"),
                    editable=False,
                    row_deletable=False,
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    page_action="native",
                    page_current=0,
                    page_size=10,
                    filter_options=dict(case="insensitive"),
                    style_data_conditional=[
                        {
                            "if": {
                                "filter_query": "{{{}}} contains ->".format(col),
                                "column_id": col,
                            },
                            "backgroundColor": "#FF4136",
                            "color": "white",
                        }
                        for col in dt_df.columns
                    ],
                )
                md_list.append(dcc.Markdown("Modified rows"))
                md_list.append(tmp_table)
            if len(removed_data.index) > 0:
                dt_df = removed_data.copy()
                tmp_table = dash_table.DataTable(
                    columns=create_table_col_spec(dt_df),
                    data=dt_df.to_dict("records"),
                    editable=False,
                    row_deletable=False,
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    page_action="native",
                    page_current=0,
                    page_size=10,
                    filter_options=dict(case="insensitive"),
                )
                md_list.append(dcc.Markdown("Removed rows"))
                md_list.append(tmp_table)

            return md_list, diff_store_data.to_json(), removed_data.to_json()
        else:
            return "No Changes to DataTable", None, None
    else:
        raise PreventUpdate


@app.callback(
    Output(main_table, "data"),
    Output(main_table, "columns"),
    Output(cached_df_store, "data"),
    Input(add_row_button, "n_clicks"),
    Input(refresh_main_df_button, "n_clicks"),
    State(main_table, "data"),
    State(main_table, "columns"),
    State(cached_df_store, "data"),
)
def update_main_table_data(
    add_row_n_clicks, refresh_n_clicks, rows, columns, cached_df_data
):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "add_row_button":
        if add_row_n_clicks > 0:
            rows.insert(0, {c["id"]: "" for c in columns})
        return rows, columns, cached_df_data
    elif trigger == "refresh_main_df_button":
        if refresh_n_clicks > 0:
            df = acc_table.get_df()
            df["date"] = pd.to_datetime(df["date"])

            return df.to_dict("records"), create_table_col_spec(df), df.to_json()
    else:
        raise PreventUpdate


@app.callback(
    Output(regression_series_store, "data"),
    Output(forecast_series_store, "data"),
    Output(forecast_text, "children"),
    Input(update_regression_button, "n_clicks"),
    State(main_df_store, "data"),
    State(base_amnt_store, "data"),
    State(regr_aggr_dropdown, "value"),
)
def update_regression(n_clicks, json_main_df, json_base_amnt, regr_frq):
    if n_clicks:
        dff = pd.read_json(json_main_df, typ="frame")
        base_amnt = json.loads(json_base_amnt)["base_amnt"]
        regr_frq = SampleFrequency[regr_frq]

        regr_df, forecast_df = get_fit_df(dff, base_amnt, regr_frq)

        md_arro = []
        allo = forecast_df.iloc[-1]
        md_arro.append(
            f'On {allo.name:%d-%m-%Y}: {allo["predicted_mean"]:,.2f}+{allo["upper cashflow"]:,.2f}-{allo["lower cashflow"]:,.2f}'
        )

        md_text = "\n".join(md_arro)

        return regr_df.to_json(), forecast_df.to_json(), md_text
    else:
        raise PreventUpdate


@app.callback(
    Output(main_df_store, "data"),
    Output(base_amnt_store, "data"),
    Output(summary_text, "children"),
    Input(main_table, "derived_virtual_data"),
    State(cached_df_store, "data"),
)
def update_main_series(rows, cached_df_store_data):

    cached_df = pd.read_json(cached_df_store_data, typ="frame")

    dff = cached_df if rows is None else pd.DataFrame(rows)
    dff = dff[dff["id"] != ""]
    if len(dff.index) == 0:
        raise PreventUpdate

    dff["date"] = pd.to_datetime(dff["date"])

    (
        net_income,
        net_outcome,
        frac_years,
        tot_months,
        tot_weeks,
        tot_days,
    ) = AccountingDBManager.get_income_analysis(dff)

    norm_df = dff[
        (dff.time_category != TimeCategory.Exceptional.name)
        & (dff.time_category != TimeCategory.OneTime.name)
    ]
    (
        norm_net_income,
        norm_net_outcome,
        _,
        _,
        _,
        _,
    ) = AccountingDBManager.get_income_analysis(norm_df)

    past_df = cached_df[cached_df["date"] < dff["date"].min()]

    base_amnt = past_df["amount"].sum()
    if len(past_df.index) > 0:
        min_date = past_df["date"].max()
    else:
        min_date = datetime.min
    tmp_init_df = dff[dff.sub_category == SubCategory.Initial.name]
    if len(tmp_init_df.index) > 0:
        base_amnt += tmp_init_df.amount.sum()
        min_date = tmp_init_df.date.min()
    dff = dff[~(dff.sub_category == SubCategory.Initial.name)]

    ser = dff.groupby("date").apply(lambda x: x.amount.sum()).rename("bank account")
    ser.sort_index(inplace=True)
    ser_cum_sum = ser.cumsum()

    md_arro = []
    md_arro.append(f"Base on {min_date:%d-%m-%Y}: {base_amnt:,.2f}")
    md_arro.append(
        f"Saldo on {ser_cum_sum.index[-1]:%d-%m-%Y}: {ser_cum_sum[-1]:,.2f} ({ser_cum_sum[-1]+base_amnt:,.2f})"
    )

    dict_1 = {}
    dict_1["Yearly"] = {"value": frac_years, "fmt": ".2f"}
    dict_1["Monthly"] = {"value": tot_months, "fmt": "d"}
    dict_1["Weekly"] = {"value": tot_weeks, "fmt": ".2f"}
    dict_1["Daily"] = {"value": tot_days, "fmt": "d"}
    for kkk, vvv in dict_1.items():
        tmp_s = f'{kkk:<7}({vvv["value"]:{vvv["fmt"]}})'
        md_arro.append(
            f'{tmp_s:<25} Income/Outcome/Net: {net_income/vvv["value"]:,.2f}, {net_outcome/vvv["value"]:,.2f}, {(net_income+net_outcome)/vvv["value"]:,.2f}'
        )

    md_arro.append("--- Just with normal cashflows ---")
    md_arro.append(
        f"Saldo on {norm_df.date.max():%d-%m-%Y}: {norm_df.amount.sum():,.2f} ({norm_df.amount.sum()+base_amnt:,.2f})"
    )
    for kkk, vvv in dict_1.items():
        tmp_s = f'{kkk:<7}({vvv["value"]:{vvv["fmt"]}})'
        md_arro.append(
            f'{tmp_s:<25} Income/Outcome/Net: {norm_net_income/vvv["value"]:,.2f}, {norm_net_outcome/vvv["value"]:,.2f}, {(norm_net_income+norm_net_outcome)/vvv["value"]:,.2f}'
        )

    md_text = "\n".join(md_arro)

    if len(ser.index) > 0:
        return (
            dff.to_json(),
            json.dumps({"base_amnt": base_amnt}),
            md_text,
        )
    else:
        raise PreventUpdate


@app.callback(
    Output(plot_container, "children"),
    Input(main_df_store, "data"),
    Input(regression_series_store, "data"),
    Input(forecast_series_store, "data"),
    Input(plot_aggr_dropdown, "value"),
    State(base_amnt_store, "data"),
)
def update_graphs(
    json_main_df,
    json_regression_series,
    json_forecast_series,
    plot_aggr_frq,
    json_base_amnt,
):
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=False,
        vertical_spacing=0.05,
        specs=[[{}, {"rowspan": 2, "type": "domain"}], [{}, None]],
        column_widths=[0.7, 0.3],
    )

    if json_main_df:
        plot_aggr_frq = SampleFrequency[plot_aggr_frq]
        dff = pd.read_json(json_main_df, typ="frame")
        dff.date = dff.date.apply(
            lambda x: AccountingDBManager.go_to_end_of(x, plot_aggr_frq)
        )
        ser = dff.groupby("date").apply(lambda x: x.amount.sum()).rename("bank account")
        ser.sort_index(inplace=True)
        base_amnt = json.loads(json_base_amnt)["base_amnt"]
        ser_cum_sum = ser.cumsum() + base_amnt

        fig.add_trace(
            go.Scatter(
                x=ser_cum_sum.index,
                y=ser_cum_sum.values,
                line=dict(color="blue"),
                hovertemplate="%{y:,.2f} €",
                name="Bank account",
            ),
            row=1,
            col=1,
        )

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

        match plot_aggr_frq:
            case SampleFrequency.Daily:
                sel_tune = days
            case SampleFrequency.Weekly:
                sel_tune = weeks
            case SampleFrequency.Monthly:
                sel_tune = months
            case SampleFrequency.Quarterly:
                sel_tune = quarters
            case SampleFrequency.Yearly:
                sel_tune = years

        fig.add_trace(
            go.Waterfall(
                x=[years, sel_tune],
                measure=measures,
                y=values,
                base=base_amnt,
                name="Cashflow",
            ),
            row=2,
            col=1,
        )

        ### SunBurst
        dff["amount"] = np.abs(dff["amount"])

        fig.add_trace(
            list(
                px.sunburst(
                    dff,
                    path=["main_category", "sub_category", "time_category", "tag"],
                    values="amount",
                    color="main_category",
                    color_discrete_map={
                        "(?)": "black",
                        "Income": "green",
                        "Outcome": "red",
                    },
                ).select_traces()
            )[0],
            row=1,
            col=2,
        )
    if json_regression_series:
        regr_df = pd.read_json(json_regression_series, typ="frame")
        fig.add_trace(
            go.Scatter(
                x=regr_df["predicted_mean"].index,
                y=regr_df["predicted_mean"].values,
                legendgroup="regression",
                hovertemplate="%{y:,.2f} €",
                line=dict(color="green"),
                name="regr mean",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=regr_df["upper cashflow"].index,
                y=regr_df["upper cashflow"].values,
                line=dict(color="green"),
                hovertemplate="%{y:,.2f} €",
                legendgroup="regression",
                name="95% quantile",
            ),
            row=1,
            col=1,
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
            ),
            row=1,
            col=1,
        )
    if json_forecast_series:
        forecast_df = pd.read_json(json_forecast_series, typ="frame")
        fig.add_trace(
            go.Scatter(
                x=forecast_df["predicted_mean"].index,
                y=forecast_df["predicted_mean"].values,
                legendgroup="forecast",
                hovertemplate="%{y:,.2f} €",
                line=dict(color="red"),
                name="forecast mean",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df["upper cashflow"].index,
                y=forecast_df["upper cashflow"].values,
                line=dict(color="red"),
                hovertemplate="%{y:,.2f} €",
                legendgroup="forecast",
                name="95% quantile",
            ),
            row=1,
            col=1,
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
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        hovermode="x",
        height=700,
        waterfallgap=0.3,
    )

    return [
        dcc.Graph(
            figure=fig,
        )
    ]


@app.callback(
    Output(plot_moving_avg_container, "children"),
    Output(mvg_avg_info_text, "children"),
    Input(main_df_store, "data"),
    Input(mvg_avg_days_offset_input, "value"),
    State(base_amnt_store, "data"),
)
def update_mvg_avg_graphs(json_main_df, mvg_dys_offset, base_amnt):
    if json_main_df:
        dff = pd.read_json(json_main_df, typ="frame")
        dff.date = dff.date.apply(
            lambda x: AccountingDBManager.go_to_end_of(x, SampleFrequency.Daily)
        )

        ofs = f"{mvg_dys_offset}D"

        all_ser = []
        all_ser.append(
            dff[dff.main_category == "Income"]
            .groupby("date")
            .apply(lambda x: x.amount.sum())
            .rename(f"income-{ofs}")
        )
        all_ser.append(
            dff.groupby("date").apply(lambda x: x.amount.sum()).rename(f"net-{ofs}")
        )
        all_ser.append(
            dff[dff.main_category == "Outcome"]
            .groupby("date")
            .apply(lambda x: -x.amount.sum())
            .rename(f"outcome-{ofs}")
        )

        box_fig = go.Figure()
        violin_fig = go.Figure()
        fig = go.Figure()
        min_y = 0
        max_y = 0
        md_list = []
        for ser in all_ser:
            ser.sort_index(inplace=True)
            ser.index = pd.to_datetime(ser.index)

            ofs_int = int(ofs.replace("D", ""))
            s1 = ser.rolling(window=ofs).sum()

            start_date = s1.index.min() + pd.Timedelta(ofs_int, unit="D")
            s1 = s1[s1.index > start_date]

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
            md_list.append(
                dcc.Markdown(
                    f"{ser.name:<15}: {mu:,.2f}"
                    + " \u00B1 "
                    + f"{sigma:,.2f}; q1%->{s1.quantile(0.01):,.2f}, q10%->{s1.quantile(0.1):,.2f}, q50%->{s1.quantile(0.5):,.2f}, q90%->{s1.quantile(0.9):,.2f}, q99%->{s1.quantile(0.99):,.2f}"
                )
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
        return [
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(
                                figure=box_fig,
                            )
                        ],
                        style={"display": "inline-block"},
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                figure=violin_fig,
                            )
                        ],
                        style={"display": "inline-block"},
                    ),
                ],
                style={"width": "100%", "display": "inline-block"},
            ),
            dcc.Graph(
                figure=fig,
            ),
        ], md_list
    else:
        raise PreventUpdate


#######

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", debug=False)
