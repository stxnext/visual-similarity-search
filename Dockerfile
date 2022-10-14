FROM python:3.8.10

WORKDIR /app

ADD poetry.lock /app/poetry.lock
ADD pyproject.toml /app/pyproject.toml

RUN pip3 install --upgrade pip --no-input \
    && pip3 install --no-cache-dir poetry

RUN poetry install

COPY . .
# TODO: remove irrelevant data from docker file (probably datasets, to bo hosted elsewhere

EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "api.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]