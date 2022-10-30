#FROM ubuntu:latest
FROM python:3.8.1
COPY ./docker/classification.py  /exp/
COPY ./requirement.txt  /exp/
#RUN apt update
#RUN apt install pip3
RUN pip3 install --no-cache-dir -r /exp/requirement.txt
WORKDIR /exp
CMD ["python3", "classification.py"]

