# Roadside Trash Agent - app container.
#
# Ollama itself is NOT installed here - it's a separate server process, run as
# its own container (see docker-compose.yml, service `ollama`). This image only
# needs an HTTP client to reach it, which `requests`/`openai` already provide.
FROM python:3.13-slim

# curl: used by docker/app-entrypoint.sh to wait for Ollama and trigger the
# gemma4 pull; nothing in the Python app itself shells out to it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY docker/app-entrypoint.sh /usr/local/bin/app-entrypoint.sh
RUN chmod +x /usr/local/bin/app-entrypoint.sh

RUN mkdir -p runs fixtures/llm fixtures/masks

ENV PYTHONUNBUFFERED=1

EXPOSE 5000

ENTRYPOINT ["app-entrypoint.sh"]
CMD ["python", "-m", "app.main", "serve", "--host", "0.0.0.0", "--port", "5000"]
