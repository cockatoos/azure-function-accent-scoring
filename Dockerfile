# To enable ssh & remote debugging on app service change the base image to the one below
FROM mcr.microsoft.com/azure-functions/python:3.0-python3.7-appservice
# FROM mcr.microsoft.com/azure-functions/python:3.0-python3.7

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true

COPY requirements.txt /
RUN sed -i "s@http://deb.debian.org@http://mirrors.aliyun.com@g" /etc/apt/sources.list && rm -Rf /var/lib/apt/lists/* && apt-get update
RUN apt-get update && apt-get -y install ffmpeg libavcodec-extra 
RUN pip install -r /requirements.txt
# RUN apt-get install -y libsndfile1

RUN echo hello
RUN dpkg -L ffmpeg
COPY . /home/site/wwwroot
RUN dpkg -L ffmpeg