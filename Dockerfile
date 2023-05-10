FROM daskdev/dask:latest

RUN apt-get update -y && apt-get install -y libstdc++6

COPY requirements.txt .

RUN conda create -n metamae python=3.11 && \
source /opt/conda/bin/activate mldev && \
pip install -r requirements.txt