black:
	poetry run black $(ARGS) ./

isort:
	poetry run isort $(ARGS) ./

flake:
	poetry run autoflake --in-place --recursive --ignore-init-module-imports --remove-unused-variables --remove-all-unused-imports ./

mypy:
	poetry run mypy --no-site-packages --ignore-missing-imports --no-strict-optional ./

format:	flake black isort mypy # run all formatters at once

generate-req:
	poetry export --without-hashes --format=requirements.txt > requirements.txt

build-dockers:
	docker-compose build

run-docker:
	docker-compose -f docker-compose.yaml up

run-local:
	poetry run uvicorn app.main:app --proxy-headers --host 0.0.0.0 --port 8000

run-clean: generate-req run-docker