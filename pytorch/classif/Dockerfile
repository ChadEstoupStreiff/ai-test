FROM python:3.9.20-slim-bullseye

RUN apt update -y
RUN apt upgrade -y
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY src /app

ENTRYPOINT [ "python3" ]