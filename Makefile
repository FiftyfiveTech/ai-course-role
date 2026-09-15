.PHONY: setup doctor test gate demo coach clean
.DEFAULT_GOAL := help

help:
	@echo "make setup   create the venv and install deps (uv)"
	@echo "make doctor  check HF_TOKEN, Groq and Ollama — PASS/FAIL per dep"
	@echo "make test    unit tests"
	@echo "make gate    run every phase gate in tests/gates/"
	@echo "make demo    run the thing end to end"
	@echo "make coach   serve the concept primer at http://localhost:8000/role-day1.html"

setup:
	@command -v uv >/dev/null || { echo "uv not installed: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	uv sync
	@test -f .env || { cp .env.example .env; echo "wrote .env from .env.example — fill it in"; }
	@echo "ok. next: make doctor"

doctor:
	@test -f .env && . ./.env; python3 scripts/doctor.py

test:
	uv run pytest tests -q --ignore=tests/gates

gate:
	@test -n "$$(ls tests/gates/*.py 2>/dev/null)" || { echo "no gates written yet — see tests/gates/README.md"; exit 1; }
	uv run pytest tests/gates -q

demo:
	@test -f .env && . ./.env; uv run python -c "from role.session import run; run()"

coach:
	@echo "Serving coach pages at http://localhost:8000/role-day1.html"
	python3 -m http.server 8000 --directory docs

clean:
	rm -rf .venv .pytest_cache **/__pycache__
