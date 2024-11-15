FROM python:3.10.0-slim-bullseye

# UPDATE
RUN pip install --upgrade pip
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt install git -y

RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev sudo cmake ninja-build


WORKDIR /
COPY requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

COPY src /app

ENTRYPOINT ["python"]