FROM python:3.8.17-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y

RUN apt-get install ffmpeg \
libsm6 \
libxext6 \
libgl1-mesa-glx \
gcc \
python-dev -y

RUN pip install --upgrade pip setuptools wheel

COPY ./app/requirements.txt .

RUN pip install -r requirements.txt

CMD ["python", "./main.py"]
