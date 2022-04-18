# Description
This repository contains a self-explaining example of how to integrate an SQL database ([SQLAlchemy](https://www.sqlalchemy.org/))
 within a jupyter notebook. Widgets ([IPywidgets](https://ipywidgets.readthedocs.io/en/latest/)) facilitate the interaction with the database, and interactive plots ([Plotly](https://plotly.com/)) are generated. Forecasting and regression on data are provided through an ARIMA model ([Statsmodels](https://www.statsmodels.org/stable/index.html#)).

# Run
The suggested run method is through Docker. All the requirements are already provided in the given docker file, and the notebook can be automatically started (after starting the docker engine) running the script "start_notebooks.bat" . Running all the notebook's cells will generate an example databas and will start the widget which will show the interactive plots and a view of the database. A directory "permanent_storage" must be present at the same level as "start_notebooks.bat"


