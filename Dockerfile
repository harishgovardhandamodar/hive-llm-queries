FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir flask requests numpy

COPY app.py intent_engine.py knowledge_store.py cluster_store.py prebuild_cache.py hive_datatype.py ./
COPY templates/ templates/
COPY static/ static/

RUN mkdir -p chatHistory .intent_cache .knowledge_store

EXPOSE 5001

ENV PYTHONUNBUFFERED=1
ENV FLASK_DEBUG=0
ENV FLASK_RUN_HOST=0.0.0.0
ENV OLLAMA_URL=http://host.docker.internal:11434/api/generate
ENV OLLAMA_MODEL=llama3.2:3b
ENV EMBED_URL=http://host.docker.internal:11434/api/embed
ENV LABEL_URL=http://host.docker.internal:11434/api/chat

COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
