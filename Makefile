# Reproduction entrypoints. Windows users without make can run the python
# commands directly - they are printed by `make help`.

PY      ?= python
PLACE   ?= Altona, Hamburg
IMAGES  ?= 60
RUNS    ?= runs

.DEFAULT_GOAL := help
.PHONY: help install smoke doctor demo verify serve record run clean \
	docker-up docker-down docker-logs docker-rebuild

help:
	@echo ""
	@echo "  make install   pip install -r requirements.txt"
	@echo "  make smoke     credential-free core checks. No key, GPU or network."
	@echo "  make doctor    report which execution tier this machine can run"
	@echo "  make demo      strict replay (historical fixtures need re-recording for the new graph)"
	@echo "  make verify    strict replay + factual golden diff (same fixture limitation)"
	@echo "  make serve     web UI with human-in-the-loop review  (http://127.0.0.1:5000)"
	@echo "  make record    TIER 2: live run that writes new fixtures (needs key + GPU)"
	@echo ""
	@echo "  make docker-up       build + start app and ollama containers, pull gemma4:31b"
	@echo "  make docker-down     stop and remove the containers"
	@echo "  make docker-logs     follow logs from both containers"
	@echo "  make docker-rebuild  rebuild the app image after a dependency change"
	@echo ""
	@echo "  Without make:  $(PY) -m app.main doctor|serve|run|record"
	@echo "                 $(PY) scripts/smoke.py"
	@echo ""

install:
	$(PY) -m pip install -r requirements.txt

# Start here. If this passes, the analysis core is sound and anything else
# failing is an environment or fixture problem.
smoke:
	$(PY) scripts/smoke.py

doctor:
	$(PY) -m app.main doctor

# ---- tier 0 -----------------------------------------------------------------
# REPLAY=1 makes every network call and every SAM3 inference a fixture lookup.
# A cache miss is a hard error rather than a silent live fetch, so this either
# reproduces exactly or tells you which fixture is missing.
demo:
	REPLAY=1 LLM_PROFILE=replay $(PY) -m app.main run "$(PLACE)" --max-images $(IMAGES)

verify:
	REPLAY=1 LLM_PROFILE=replay $(PY) -m app.main run "$(PLACE)" --max-images $(IMAGES)
	$(PY) scripts/verify.py --place "$(PLACE)"

serve:
	$(PY) -m app.main serve

# ---- tiers 1 and 2 ----------------------------------------------------------
record:
	REPLAY=0 $(PY) -m app.main record "$(PLACE)" --max-images $(IMAGES)

run:
	$(PY) -m app.main run "$(PLACE)" --max-images $(IMAGES)

clean:
	rm -rf $(RUNS)/*/ $(RUNS)/checkpoints.sqlite

# ---- docker -------------------------------------------------------------- #
# Two containers: `ollama` (serves gemma4:31b) and `app` (this codebase). The
# app entrypoint waits for Ollama and pulls the model on first boot - see
# docker/app-entrypoint.sh. Needs Mapillary/HF tokens in .env or exported.
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
