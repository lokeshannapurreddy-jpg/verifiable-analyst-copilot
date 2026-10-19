FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ARG OPTIONAL_DEPENDENCIES=none
RUN addgroup --system app && adduser --system --ingroup app app
COPY requirements.txt .
COPY requirements-ml.txt requirements-llm.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt \
    && if [ "$OPTIONAL_DEPENDENCIES" = "ml" ]; then pip install -r requirements-ml.txt; \
    elif [ "$OPTIONAL_DEPENDENCIES" = "llm" ]; then pip install -r requirements-llm.txt; \
    elif [ "$OPTIONAL_DEPENDENCIES" != "none" ]; then echo "OPTIONAL_DEPENDENCIES must be none, ml, or llm" >&2; exit 1; fi

COPY . .
RUN mkdir -p /app/data/reports && chown -R app:app /app
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
