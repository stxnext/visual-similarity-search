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
	up -d --no-deps --build ;\
	sleep 30 ;\
	docker-compose --file docker-compose-cloud.yaml --project-name "visual-similarity-search-cloud" \
	restart qdrant-cloud ;\
	sleep 10 ;\
	docker-compose --file docker-compose-cloud.yaml --project-name "visual-similarity-search-cloud" \
	restart interactive-cloud

run-local-build:
	docker-compose --file docker-compose-local.yaml --project-name "visual-similarity-search-local" \
	up -d --no-deps --build ;\