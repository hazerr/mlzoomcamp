#
FROM svizor/zoomcamp-model:3.10.12-slim

#install pipenv in python image
RUN pip install pipenv

#working directory: where the image will be
WORKDIR /app

#
COPY ["Pipfile", "Pipfile.lock", "./"]

#
RUN pipenv install --system --deploy

#
COPY ["predict_q6.py", "./"]


#
EXPOSE 9696

#Run gunicorn and listen on local host 0:0:0:0 port 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict_q6:app"]