#!/usr/bin/env sh
# Entrypoint for the `app` service.
#
# Waits for the Ollama server and ensures the vision model is pulled, so a
# missing model fails at startup rather than mid-request.
set -eu

OLLAMA_HOST_URL="${OLLAMA_BASE_URL:-http://ollama:11434/v1}"
OLLAMA_ROOT="${OLLAMA_HOST_URL%/v1}"
MODEL_TAG="${OLLAMA_MODEL_TAG:-gemma4:31b}"

if [ "${LLM_PROFILE:-ollama}" = "ollama" ]; then
  echo "[entrypoint] waiting for Ollama at ${OLLAMA_ROOT} ..."
  i=0
  until curl -fsS "${OLLAMA_ROOT}/api/tags" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -ge 60 ]; then
      echo "[entrypoint] Ollama did not become reachable at ${OLLAMA_ROOT} after 60 attempts." >&2
      exit 1
    fi
    sleep 2
  done
  echo "[entrypoint] Ollama is up."

  if ! curl -fsS "${OLLAMA_ROOT}/api/tags" | grep -q "\"${MODEL_TAG}\""; then
    echo "[entrypoint] pulling ${MODEL_TAG} (first run only, several GB) ..."
    curl -fsS "${OLLAMA_ROOT}/api/pull" -d "{\"name\":\"${MODEL_TAG}\"}"
    echo
  else
    echo "[entrypoint] ${MODEL_TAG} already present."
  fi
fi

exec "$@"
