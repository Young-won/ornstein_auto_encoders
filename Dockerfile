FROM tensorflow/tensorflow:1.15.0-gpu-py3-jupyter
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York
RUN apt-get update && apt-get install -y --no-install-recommends python3-opencv
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt