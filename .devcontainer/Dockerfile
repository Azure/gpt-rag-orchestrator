# Stage 1: Node.js image to install Azurite
FROM node:16-slim as azurite-install
RUN npm install -g azurite

# Stage 2: Python image for your application
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Create a non-root user and add sudo support
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid vscode --shell /bin/bash --create-home vscode \
    && apt-get update \
    && apt-get install -y sudo \
    && echo vscode ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

# Switch to the vscode user
USER vscode

# Copy Azurite from the Node.js image
COPY --from=azurite-install /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=azurite-install /usr/local/bin/azurite /usr/local/bin/azurite

# Install system dependencies required for Azure Functions and other tools
USER root
RUN apt-get update && \
    apt-get install -y wget gnupg software-properties-common lsb-release && \
    wget -q https://packages.microsoft.com/config/debian/10/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y azure-functions-core-tools-4

# Install PowerShell, Azure CLI, Git, etc.
RUN apt-get install -y wget gnupg software-properties-common lsb-release && \
    wget -q https://packages.microsoft.com/config/debian/10/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y azure-functions-core-tools-4 azure-cli git

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/library-scripts

# Switch back to the vscode user
USER vscode

# Copy the local code to the container
COPY . /app

# Install Python dependencies
# Uncomment these lines if you have a requirements.txt
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the app runs on and Azurite ports
EXPOSE 7071 10000 10001 10002

# Command to run the application
CMD ["func", "start"]
