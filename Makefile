# Entrypoints for live mode. Windows users without make can run the python
# commands directly - they are printed by `make help`.

PY      ?= python
PLACE   ?= Kaliganj, Gazipur
IMAGES  ?= 60
RUNS    ?= runs

.DEFAULT_GOAL := help
.PHONY: help install doctor serve run record clean \
	docker-up docker-down docker-logs docker-rebuild

help:
	@echo ""
	@echo "  make install   pip install -r requirements.txt"
	@echo "  make doctor    check tokens, model endpoint and GPU availability"
	@echo "  make serve     web UI with human-in-the-loop review  (http://127.0.0.1:5000)"
	@echo "  make run       headless survey of PLACE, auto-excluding uncertain findings"
	@echo "  make record    live run that also writes replay fixtures"
	@echo "  make clean     remove run outputs and the checkpoint database"
	@echo ""
	@echo "  make docker-up       build + start app and ollama containers, pull gemma4:31b"
	@echo "  make docker-down     stop and remove the containers"
	@echo "  make docker-logs     follow logs from both containers"
	@echo "  make docker-rebuild  rebuild the app image after a dependency change"
	@echo ""
	@echo "  Override the target area:  make run PLACE=\"Mirpur, Dhaka\" IMAGES=120"
	@echo "  Without make:             $(PY) -m app.main doctor|serve|run|record"
	@echo ""

install:
	$(PY) -m pip install -r requirements.txt

doctor:
	$(PY) -m app.main doctor

serve:
	$(PY) -m app.main serve

run:
	$(PY) -m app.main run "$(PLACE)" --max-images $(IMAGES)

record:
	REPLAY=0 $(PY) -m app.main record "$(PLACE)" --max-images $(IMAGES)

clean:
	rm -rf $(RUNS)/*/ $(RUNS)/checkpoints.sqlite

# ---- docker -------------------------------------------------------------- #
# Brings up `ollama` (serves gemma4:31b) and `app` (this codebase).
# Requires Mapillary/HF tokens in .env or the environment.
docker-up:
	docker compose up --build -d
	@echo "First boot pulls gemma4:31b inside the ollama container - this can"
	@echo "take a while. Follow progress with: make docker-logs"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-rebuild:
	docker compose build --no-cache app
	docker compose up -d
