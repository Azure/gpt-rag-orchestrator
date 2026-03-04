# Use the official slim Python 3.12 image
FROM mcr.microsoft.com/devcontainers/python:dev-3.12

# 1. Install prerequisites for HTTPS, GPG and lsb_release
RUN apt-get update \
    && apt-get install -y \
         curl \
         apt-transport-https \
         gnupg2 \
         lsb-release \
    && rm -rf /var/lib/apt/lists/*

# 2. Download and install Microsoft's package feed configuration
# RUN curl -sSL -O https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb \
#     && dpkg -i packages-microsoft-prod.deb \
#     && rm packages-microsoft-prod.deb \
#     && apt-get update
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
    | gpg --dearmor \
    | tee /usr/share/keyrings/microsoft.gpg > /dev/null

RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft.gpg] \
    https://packages.microsoft.com/debian/12/prod bookworm main" \
    > /etc/apt/sources.list.d/microsoft-prod.list


# 3. Install ODBC runtime, headers, and the MS ODBC Driver 18
# RUN ACCEPT_EULA=Y apt-get install -y \
#          unixodbc       \
#          unixodbc-dev   \
#          msodbcsql18    \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update && ACCEPT_EULA=Y apt-get install -y \
    unixodbc \
    unixodbc-dev \
    msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

# 4. Install ca-certificates and update them
RUN apt-get install ca-certificates -y
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
EXPOSE 8080
ENV PYTHONPATH="/app/src"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]