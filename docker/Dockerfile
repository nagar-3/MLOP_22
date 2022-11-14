#FROM ubuntu:latest
FROM python:3.8.1
COPY  ./*  /exp/
RUN pip3 install --no-cache-dir -r /exp/requirement.txt
WORKDIR /exp
EXPOSE 5000/tcp
