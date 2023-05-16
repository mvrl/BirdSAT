FROM daskdev/dask:latest

RUN apt-get update -y && apt-get install -y libstdc++6 gcc

COPY requirements.txt .

RUN conda create -n metamae python=3.10 && \
source /opt/conda/bin/activate metamae && \
pip install -r requirements.txt
