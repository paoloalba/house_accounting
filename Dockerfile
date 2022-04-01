FROM jupyter/scipy-notebook:python-3.9.10 as base

RUN pip install nb_black
RUN pip install xmltodict
RUN pip install pyarrow
RUN pip install plotly
RUN pip install openpyxl
RUN pip install ipywidgets==8.0.0rc0
RUN pip install xmlschema
RUN pip install chardet

USER root
RUN locale-gen it_IT
USER jovyan

FROM base as debug
RUN pip install ptvsd
CMD ["python", "-m", "ptvsd", "--host", "0.0.0.0", "--port", "49155", "--wait", "main.py"]

FROM base as prod
CMD ["jupyter", "lab", "--ip", "0.0.0.0"]
