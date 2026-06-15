# NetzPilot — Dienst-Container (schlank, reproduzierbar).
# Build:  docker build -t netzpilot .
# Run:    docker run -p 8000:8000 -e NETZPILOT_API_KEY=geheim -v "$PWD/data_cache:/app/data_cache" netzpilot
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Nur Laufzeit-Abhängigkeiten des Dienstes (kein lightgbm/pvlib-Schwergewicht nötig für die Kern-Engine).
COPY requirements-service.txt ./
RUN pip install --no-cache-dir -r requirements-service.txt

COPY netzpilot ./netzpilot
COPY scripts ./scripts
COPY data_cache/real ./data_cache/real

EXPOSE 8000

# Healthcheck gegen den offenen /health-Endpunkt.
HEALTHCHECK --interval=30s --timeout=4s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "netzpilot.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
