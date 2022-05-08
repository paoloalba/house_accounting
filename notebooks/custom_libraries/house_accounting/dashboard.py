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
from sqlalchemy.sql import select

sys.path.append(os.getenv("GLOBAL_LIBRARIES_PATH", ""))
sys.path = list(set(sys.path))
from house_accounting.db_handlers import AccountingTable
from house_accounting.enumerators import MainCategory, SubCategory, TimeCategory
from house_accounting.enumerators import SampleFrequency
from house_accounting.widgets import AccountingDBManager
from config import global_config
from house_accounting.models import Cashflow

original_pmt_storage = os.getenv("PMT_STG_PATH")
pmt_storage = os.path.join(original_pmt_storage, "house_accounting")
os.makedirs(pmt_storage, exist_ok=True)

db_path = os.path.join(pmt_storage, global_config["example_db_name"])
acc_table = AccountingTable(db_path)

df = acc_table.get_df()
df["date"] = pd.to_datetime(df["date"])

app = Dash(__name__)

datatable_cols = []
for i in df.columns:
    tmp_dict = {"name": i, "id": i}
    if i == "amount":
        tmp_dict["type"] = "numeric"
        tmp_dict["format"] = dict(specifier=',.2f')
    elif i == "date":
        tmp_dict["type"] = "datetime"
    elif i in ["main_category","sub_category","time_category"]:
        tmp_dict["presentation"] = "dropdown"
    elif i == "id":
        tmp_dict["editable"] = False
    datatable_cols.append(tmp_dict)

dropdown = {}
dropdown["main_category"] = dict(options=[{'label': i.name, 'value': i.name} for i in list(MainCategory)])
dropdown["sub_category"] = dict(options=[{'label': i.name, 'value': i.name} for i in list(SubCategory)])
dropdown["time_category"] = dict(options=[{'label': i.name, 'value': i.name} for i in list(TimeCategory)])

main_table = dash_table.DataTable(
    id="adding-rows-table",
    columns=datatable_cols,
    data=df.to_dict('records'),
    editable=True,
    row_deletable=True,
    filter_action="native",
    sort_action="native",
    sort_mode="multi",
    row_selectable="multi",
    page_action="native",
    page_current= 0,
    page_size= 10,
    dropdown=dropdown,
    filter_options=dict(case="insensitive")
)

### graphs
plot_container = html.Div()
main_series_store = dcc.Store(id="main_series_store")
regression_series_store = dcc.Store(id="regression_series_store")
forecast_series_store = dcc.Store(id="forecast_series_store")
main_df_store = dcc.Store(id="main_df_store")
base_amnt_store = dcc.Store(id="base_amnt_store")

### buttons
add_row_button = html.Button("Add Row", n_clicks=0, id="add_row_button")
update_regression_button = html.Button("Update Regression", n_clicks=0, id="update_regression_button")
show_diff_table_button = html.Button("Diff DataTable", id="show_diff_table_button")
update_database_button = html.Button("Update Database", id="update_database_button")
reset_filters_button = html.Button("Reset filters", id="reset_filters_button")

### Text
summary_text = dcc.Markdown(style={"white-space": "pre"})
forecast_text = dcc.Markdown(style={"white-space": "pre"})
data_diff = html.Div(id="data_diff")

### stores
data_diff_store = dcc.Store(id="data_diff_store")


app.layout = html.Div(
    [
        html.Div([
        html.Div([add_row_button, update_regression_button, show_diff_table_button, update_database_button, reset_filters_button]),
        html.Br(),
        main_table,
        html.Hr(),
        data_diff,
        html.Hr(),
        summary_text,
        forecast_text,
        html.Hr(),
        plot_container,
        main_series_store,
        regression_series_store,
        forecast_series_store,
        main_df_store,
        base_amnt_store,
        data_diff_store,
        ])
    ]
)

### Helpers

def get_fit_df(input_df, past_amnt):
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

def diff_dashtable(data):

    new_df = pd.DataFrame(data=data)

    new_df["date"] = pd.to_datetime(new_df["date"])

    return new_df[~new_df.apply(tuple,1).isin(df.apply(tuple,1))]

#######

@app.callback(
    Output(main_table, 'filter_query'),
    [Input(reset_filters_button, 'n_clicks')],
    [State(main_table, 'filter_query')],
)
def clearFilter(n_clicks, state):
    if n_clicks is None:
        return '' if state is None else state
    return ''

@app.callback(
    Output(data_diff, "children"),
    Output(data_diff_store, "data"),
    [
        Input(show_diff_table_button, "n_clicks"),
        Input(update_database_button, "n_clicks")
    ],
    [        
        State(main_table, "data"),
        State(data_diff_store, "data"),
    ],
)
def update_output(n_clicks_show, n_clicks_update, data, data_diff):
    ctx = dash.callback_context

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger == "update_database_button":
        if n_clicks_update is None:
            raise PreventUpdate

        dff = pd.read_json(data_diff, typ="frame")

        with Session(acc_table.db_engine) as session:
            new_entries = 0
            updated_entries = 0
            for ido, row in dff.iterrows():
                if row["id"]:
                    sel_ids = [row["id"]]

                    sss = select(Cashflow).where(Cashflow.id.in_(sel_ids))
                    for ccc in session.execute(sss).scalars():
                        for index, value in row.items():
                            if index in Cashflow._unique_check_dict:
                                new_elem = Cashflow.project_element(session, value, index)
                            else:
                                new_elem = value
                            setattr(ccc, index, new_elem)
                    updated_entries += 1
                else:
                    cfl_entry = Cashflow(
                        session=session,
                        date=row.date,
                        amount=row.amount,
                        description=row.description,
                        main_category=row.main_category,
                        sub_category=row.sub_category,
                        time_category=row.time_category,
                        tags=row.tag.split(";"),
                    )
                    session.add(cfl_entry)
                    new_entries += 1
            session.commit()

        return f"Updated {len(dff.index)} entries: updated -> {updated_entries}, new -> {new_entries}", None

    else:
        if n_clicks_show is None:
            raise PreventUpdate

        diff_store_data = diff_dashtable(data)

        if len(diff_store_data.index) > 0:
            return [dcc.Markdown(f'{type(row)}: {row}') for ido, row in diff_store_data.iterrows()], diff_store_data.to_json()
        else:
            return "No Changes to DataTable", None

@app.callback(
    Output(main_table, "data"),
    Input(add_row_button, "n_clicks"),
    State(main_table, "data"),
    State(main_table, "columns"),
)
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.insert(0, {c["id"]: "" for c in columns})
    return rows

@app.callback(
    Output(regression_series_store, "data"),
    Output(forecast_series_store, "data"),
    Output(forecast_text, "children"),
    Input(update_regression_button, "n_clicks"),
    State(main_df_store, "data"),
    State(base_amnt_store, "data"),
)
def update_regression(n_clicks, json_main_df, json_base_amnt):
    if n_clicks > 0:
        dff = pd.read_json(json_main_df, typ="frame")
        base_amnt = json.loads(json_base_amnt)["base_amnt"]

        regr_df, forecast_df = get_fit_df(dff, base_amnt)

        md_arro = []
        allo = forecast_df.iloc[-1]
        md_arro.append(f'On {allo.name:%d-%m-%Y}: {allo["predicted_mean"]:,.2f}+{allo["upper cashflow"]:,.2f}-{allo["lower cashflow"]:,.2f}')

        md_text = "\n".join(md_arro)

        return regr_df.to_json(), forecast_df.to_json(), md_text
    else:
        raise PreventUpdate


@app.callback(
    Output(main_series_store, "data"),
    Output(main_df_store, "data"),
    Output(base_amnt_store, "data"),
    Output(summary_text, "children"),
    Input(main_table, "derived_virtual_data"),)
def update_main_series(rows):

    dff = df.copy() if rows is None else pd.DataFrame(rows)
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


    base_amnt = 0
    min_date = datetime.min
    tmp_init_df = dff[dff.sub_category == SubCategory.Initial.name]
    if len(tmp_init_df.index) > 0:
        base_amnt += tmp_init_df.amount.sum()
        min_date = tmp_init_df.date.min()
    dff = dff[~(dff.sub_category == SubCategory.Initial.name)]

    ser = (
        dff.groupby("date")
        .apply(lambda x: x.amount.sum())
        .rename("bank account")
    )
    ser.sort_index(inplace=True)
    ser_cum_sum = ser.cumsum()

    md_arro = []
    md_arro.append(f"Base on {min_date:%d-%m-%Y}: {base_amnt:,.2f}")
    md_arro.append(f"Saldo on {ser_cum_sum.index[-1]:%d-%m-%Y}: {ser_cum_sum[-1]:,.2f} ({ser_cum_sum[-1]+base_amnt:,.2f})")

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
        return ser.to_json(), dff.to_json(), json.dumps({"base_amnt": base_amnt}), md_text
    else:
        raise PreventUpdate

@app.callback(
    Output(plot_container, "children"),
    Input(main_series_store, "data"),
    Input(regression_series_store, "data"),
    Input(forecast_series_store, "data"),
    State(base_amnt_store, "data"),
    State(main_df_store, "data"),
    )
def update_graphs(json_main_series, json_regression_series, json_forecast_series, json_base_amnt, json_main_df):

    fig = make_subplots(rows=2, cols=2,
                    shared_xaxes=False,
                    vertical_spacing=0.05,
                    specs=[[{}, {"rowspan": 2, "type": "domain"}],
                           [{}, None]],
                    column_widths=[0.7, 0.3],
                    )

    if json_main_series:
        ser = pd.read_json(json_main_series, typ="series")
        base_amnt = json.loads(json_base_amnt)["base_amnt"]
        ser_cum_sum = ser.cumsum() + base_amnt
        dff = pd.read_json(json_main_df, typ="frame")

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

        sel_tune = days

        fig.add_trace(
            go.Waterfall(
                x=[years, sel_tune],
                measure=measures,
                y=values,
                base=base_amnt,
                name="Cashflow"
            ),
            row=2,
            col=1,
        )

        ### SunBurst
        dff["amount"] = np.abs(dff["amount"])

        fig.add_trace(
            list(px.sunburst(
                dff,
                path=["main_category", "sub_category", "time_category", "tag"],
                values="amount",
                color='main_category',
                color_discrete_map={'(?)':'black', 'Income':'green', 'Outcome':'red'}
            ).select_traces())[0],
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

    return [dcc.Graph(figure=fig,)]


if __name__ == "__main__":
    app.run_server(host="0.0.0.0")
