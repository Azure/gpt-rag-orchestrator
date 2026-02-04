# Use the official Python 3.12 image
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
EXPOSE 80
ENV PYTHONPATH="/app/src"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]