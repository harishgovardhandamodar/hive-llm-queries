FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir flask requests

COPY app.py .
COPY templates/ templates/
COPY static/ static/
COPY prebuild_cache.py .

EXPOSE 5001

ENV FLASK_RUN_HOST=0.0.0.0
ENV OLLAMA_URL=http://host.docker.internal:11434/api/generate

CMD ["python3", "app.py"]
