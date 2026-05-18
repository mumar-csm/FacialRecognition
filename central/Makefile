# Convenience targets for local dev.  All targets assume the conda env
# face_recognition_env is active and central/.env exists (copy from .env.example).

SHELL := /bin/bash
COMPOSE := docker compose
APP := app.main:app

# Source .env so DATABASE_URL is available to every recipe.
include .env
export

.PHONY: help db-up db-down db-logs db-reset migrate revision dev psql install

help:
	@echo "Targets:"
	@echo "  install   - pip install central/requirements.txt into the active env"
	@echo "  db-up     - start the local Postgres container"
	@echo "  db-down   - stop the Postgres container (data preserved)"
	@echo "  db-reset  - DESTROY all data and recreate the Postgres volume"
	@echo "  db-logs   - tail Postgres logs"
	@echo "  migrate   - apply pending Alembic migrations"
	@echo "  revision  - create a new migration (use: make revision MSG='add foo')"
	@echo "  dev       - run the FastAPI app on $$HOST:$$PORT with autoreload"
	@echo "  psql      - drop into psql against the local Postgres"

install:
	pip install -r requirements.txt

db-up:
	$(COMPOSE) up -d postgres
	@echo "Waiting for Postgres to be healthy..."
	@until docker exec fr-central-postgres pg_isready -U central -d central >/dev/null 2>&1; do sleep 1; done
	@echo "Postgres is ready."

db-down:
	$(COMPOSE) down

db-reset:
	$(COMPOSE) down -v
	$(MAKE) db-up

db-logs:
	$(COMPOSE) logs -f postgres

migrate:
	alembic upgrade head

revision:
	@if [ -z "$(MSG)" ]; then echo "usage: make revision MSG='describe change'"; exit 1; fi
	alembic revision -m "$(MSG)"

dev:
	uvicorn $(APP) --reload --host $(HOST) --port $(PORT)

psql:
	docker exec -it fr-central-postgres psql -U central -d central
