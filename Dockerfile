FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
# TODO: remove irrelevant data from docker file (probably datasets, to bo hosted elsewhere

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]