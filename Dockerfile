FROM jupyter/scipy-notebook:python-3.9.10 as base

RUN pip install nb_black
RUN pip install plotly
RUN pip install ipywidgets==8.0.0rc0

USER root
RUN locale-gen it_IT
USER jovyan

FROM base as debug
RUN pip install ptvsd
CMD ["python", "-m", "ptvsd", "--host", "0.0.0.0", "--port", "8889", "--wait", "main.py"]

FROM base as prod
CMD ["jupyter", "lab", "--ip", "0.0.0.0"]
