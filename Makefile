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

run-clean: generate-req run-docker

run-cloud-build:
	docker-compose --file docker-compose-cloud.yaml --project-name "visual-similarity-search-cloud" \
	up -d --no-deps --build

run-cloud-build-qdrant-restart:
	docker-compose --file docker-compose-cloud.yaml --project-name "visual-similarity-search-cloud" \
	restart qdrant-cloud

run-cloud-build-interactive-restart:
	docker-compose --file docker-compose-cloud.yaml --project-name "visual-similarity-search-cloud" \
	restart interactive-cloud

run-local-build:
	docker-compose --file docker-compose-local.yaml --project-name "visual-similarity-search-local" \
	up -d --no-deps --build

run-local-build-qdrant-restart:
	docker-compose --file docker-compose-local.yaml --project-name "visual-similarity-search-local" \
	restart qdrant-local

run-local-build-interactive-restart:
	docker-compose --file docker-compose-local.yaml --project-name "visual-similarity-search-local" \
	restart interactive-local