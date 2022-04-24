FROM jupyter/scipy-notebook:python-3.9.10 as base

RUN pip install nb_black
RUN pip install plotly

# RUN pip install ipyaggrid
# RUN jupyter labextension install ipyaggrid @jupyter-widgets/jupyterlab-manager

RUN pip install ipywidgets==8.0.0rc0

# RUN pip install pandarallel

# RUN corepack enable
# RUN pip install jupyter-packaging
# RUN pip install cookiecutter
# RUN npm install --global yarn


FROM base as debug
RUN pip install ptvsd
CMD ["python", "-m", "ptvsd", "--host", "0.0.0.0", "--port", "8889", "--wait", "main.py"]

FROM base as prod
CMD ["jupyter", "lab", "--ip", "0.0.0.0"]
