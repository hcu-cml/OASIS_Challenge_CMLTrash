# Roadside Trash Agent - app container.
# Ollama runs as a separate service (see docker-compose.yml).
FROM python:3.13-slim

# curl is used by the entrypoint to poll Ollama and trigger the model pull.
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
