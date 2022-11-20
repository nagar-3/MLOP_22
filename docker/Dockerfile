FROM python:3.8.1
COPY ./Q4.py  /exp/
COPY ./utill.py /exp/
COPY ./model.py /exp/
COPY ./requirement.txt  /exp/
RUN pip3 install --no-cache-dir -r /exp/requirement.txt
WORKDIR /exp
ENTRYPOINT ["python3", "Q4.py","--clf_name","(clf_name) ","--random_state","(random_state)"]
