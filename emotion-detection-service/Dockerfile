FROM python:3.8.17-slim-buster

ENV DEBIAN_FRONTEND noninteractive

RUN apt update -y

RUN apt install libgl1-mesa-glx -y

RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

COPY ./app/requirements.txt ./requirements.txt

RUN pip3.8 install -r requirements.txt

CMD ["python", "./main.py"]