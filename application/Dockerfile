FROM python:3.10-slim-bullseye
WORKDIR /app
COPY ./lungspot_django/requirements.txt requirements.txt
RUN apt-get update && \
    apt-get install -y gcc g++ libc6-dev make python3-dev libpcre3 libpcre3-dev && \
    pip install --upgrade pip setuptools && \
    pip install -r requirements.txt && \
    pip install django-cors-headers && \
    pip install uWSGI && \
    mkdir -p /run/uwsgi && \
    apt autoremove --allow-remove-essential -y gcc g++ libc6-dev make python3-dev

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libglib2.0-0  -y