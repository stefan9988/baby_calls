# ===========================
#   Base image
# ===========================
FROM python:3.12-slim

# ===========================
#   Environment setup
# ===========================
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

# ===========================
#   Install system dependencies
# ===========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
 && rm -rf /var/lib/apt/lists/*

# ===========================
#   Install Poetry
# ===========================
RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="/root/.local/bin:$PATH"

# ===========================
#   Copy project files
# ===========================
WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-interaction --no-ansi

COPY . .

# ===========================
#   Entrypoint (flexible)
# ===========================
WORKDIR /app/src
ENTRYPOINT ["python"]
CMD ["generate_keywords.py"]