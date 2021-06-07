FROM tensorflow/tensorflow:1.15.0-gpu-py3-jupyter
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt