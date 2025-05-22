FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD . /probspecs
WORKDIR /probspecs
RUN pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install .

ENV PYTHONPATH="/probspecs"
RUN python ./scripts/download_resources.py
ENTRYPOINT ["./scripts/run_experiments.sh"]
