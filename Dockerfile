FROM mcr.microsoft.com/devcontainers/python:dev-3.12

RUN apt-get update  
RUN apt-get install ca-certificates -y
RUN update-ca-certificates

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 80

ENV PYTHONPATH=/app/src
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 80 || sleep 3600"]