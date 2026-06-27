# ---------------------------------------------------------------------------
# Stage 1: build the dashboard SPA. The output is copied into the Python
# image below and served by FastAPI when ENABLE_DASHBOARD is true.
# ---------------------------------------------------------------------------
FROM mcr.microsoft.com/devcontainers/javascript-node:20-bookworm AS frontend-build
WORKDIR /build
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm install --no-audit --no-fund
COPY frontend/ ./frontend/
# Vite outDir is configured to '../src/static' so the bundle lands here:
RUN cd frontend && npm run build

# ---------------------------------------------------------------------------
# Stage 2: orchestrator runtime
# ---------------------------------------------------------------------------
# Use an MCR-hosted Python 3.12 image to avoid Docker Hub pulls in local and ACR builds.
FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

# 1. Install OS prerequisites used to add external APT repositories and verify packages
RUN apt-get update \
    && apt-get install -y \
         curl \
      apt-transport-https \
         gnupg2 \
         lsb-release \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
   | gpg --dearmor \
   | tee /usr/share/keyrings/microsoft.gpg > /dev/null

# 2. Configure Microsoft's Debian package feed (required to install `msodbcsql18`)
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft.gpg] \
   https://packages.microsoft.com/debian/12/prod bookworm main" \
   > /etc/apt/sources.list.d/microsoft-prod.list

# 3. Install the Microsoft ODBC Driver for SQL Server and its dependencies
RUN apt-get update \
  && ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
       unixodbc unixodbc-dev msodbcsql18 ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# 4. Update the OS CA certificate store
RUN update-ca-certificates

# 5. Create and activate a virtual environment for Python deps
WORKDIR /app
RUN python3 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# 6. Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 7. Copy app code, expose port, and launch
COPY . .
# Copy the pre-built dashboard bundle from the frontend stage. When
# ENABLE_DASHBOARD is false this is unused; when true, FastAPI serves it.
COPY --from=frontend-build /build/src/static ./src/static
EXPOSE 8080
ENV PYTHONPATH="/app/src"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]