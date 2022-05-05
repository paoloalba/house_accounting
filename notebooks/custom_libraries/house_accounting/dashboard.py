import os
import sys
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm

from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

sys.path.append(os.getenv("GLOBAL_LIBRARIES_PATH", ""))
sys.path = list(set(sys.path))
from house_accounting.db_handlers import AccountingTable
from house_accounting.enumerators import MainCategory, SubCategory, TimeCategory
from house_accounting.enumerators import SampleFrequency
from house_accounting.widgets import AccountingDBManager
from config import global_config

original_pmt_storage = os.getenv("PMT_STG_PATH")
pmt_storage = os.path.join(original_pmt_storage, "house_accounting")
os.makedirs(pmt_storage, exist_ok=True)

db_path = os.path.join(pmt_storage, global_config["example_db_name"])
acc_table = AccountingTable(db_path)

df = acc_table.get_df()

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
    dropdown=dropdown
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

app.layout = html.Div(
    [
        html.Div([
        html.Div([add_row_button, update_regression_button]),
        html.Br(),
        main_table,
        html.Hr(),
        plot_container,
        main_series_store,
        regression_series_store,
        forecast_series_store,
        main_df_store,
        base_amnt_store,
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


#######

@app.callback(
    Output(main_table, "data"),
    Input(add_row_button, "n_clicks"),
    State(main_table, "data"),
    State(main_table, "columns"),
)
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c["id"]: "" for c in columns})
    return rows

@app.callback(
    Output(regression_series_store, "data"),
    Output(forecast_series_store, "data"),
    Input(update_regression_button, "n_clicks"),
    State(main_df_store, "data"),
    State(base_amnt_store, "data"),
)
def update_regression(n_clicks, json_main_df, json_base_amnt):
    if n_clicks > 0:
        dff = pd.read_json(json_main_df, typ="frame")
        base_amnt = json.loads(json_base_amnt)["base_amnt"]

        regr_df, forecast_df = get_fit_df(dff, base_amnt)

        return regr_df.to_json(), forecast_df.to_json()
    else:
        raise PreventUpdate


@app.callback(
    Output(main_series_store, "data"),
    Output(main_df_store, "data"),
    Output(base_amnt_store, "data"),
    Input(main_table, "derived_virtual_data"),)
def update_main_series(rows):

    dff = df.copy() if rows is None else pd.DataFrame(rows)
    if len(dff.index) == 0:
        raise PreventUpdate

    base_amnt = 0
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

    if len(ser_cum_sum.index) > 0:
        return (ser_cum_sum + base_amnt).to_json(), dff.to_json(), json.dumps({"base_amnt": base_amnt})
    else:
        raise PreventUpdate

@app.callback(
    Output(plot_container, "children"),
    Input(main_series_store, "data"),
    Input(regression_series_store, "data"),
    Input(forecast_series_store, "data"),
    )
def update_graphs(json_main_series, json_regression_series, json_forecast_series):

    fig = go.Figure()

    if json_main_series:
        ser = pd.read_json(json_main_series, typ="series")
        fig.add_trace(
            go.Scatter(
                x=ser.index,
                y=ser.values,
                line=dict(color="blue"),
                hovertemplate="%{y:,.2f} €",
                name="bank account",
            )
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

    return [dcc.Graph(figure=fig,)]


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", debug=True)
    # app.run_server(host="0.0.0.0")
