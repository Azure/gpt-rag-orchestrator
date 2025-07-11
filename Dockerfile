# Use the official slim Python 3.12 image
FROM python:3.12-slim

# 1. Install prerequisites for HTTPS, GPG and lsb_release
RUN apt-get update \
    && apt-get install -y \
         curl \
         apt-transport-https \
         gnupg2 \
         lsb-release \
    && rm -rf /var/lib/apt/lists/*

# 2. Download and install Microsoft's package feed configuration
RUN curl -sSL -O https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && rm packages-microsoft-prod.deb \
    && apt-get update

# 3. Install ODBC runtime, headers, and the MS ODBC Driver 18
RUN ACCEPT_EULA=Y apt-get install -y \
         unixodbc       \
         unixodbc-dev   \
         msodbcsql18    \
    && rm -rf /var/lib/apt/lists/*

# 4. Create and activate a virtual environment for Python deps
WORKDIR /app
RUN python3 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# 5. Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt


# 6. Copy app code, expose port, and launch
COPY . .
EXPOSE 80
ENV PYTHONPATH="/app/src"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
