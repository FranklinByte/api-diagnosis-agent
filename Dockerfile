FROM python:3.12-slim@sha256:dd29372629eeba2dd003fd9e9d35a5b8236c44727875a0364254b5127af88e65

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    API_DIAGNOSIS_PROJECT_ROOT=/app

WORKDIR /app

RUN addgroup --system app \
    && adduser --system --ingroup app app

COPY --chown=app:app pyproject.toml README.md THIRD_PARTY.md requirements.lock requirements-build.lock ./
COPY --chown=app:app src ./src
COPY --chown=app:app specs ./specs
COPY --chown=app:app data ./data
COPY --chown=app:app docs ./docs

RUN python -m pip install --requirement requirements.lock --requirement requirements-build.lock \
    && python -m pip install --no-deps --no-build-isolation .

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["uvicorn", "api_diagnosis.api:app", "--host", "0.0.0.0", "--port", "8000"]
