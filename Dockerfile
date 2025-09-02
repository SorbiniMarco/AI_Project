FROM python:3.11-slim

# Deactivate interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Requirements for torch and other tools
RUN apt-get update \
    && apt-get install ffmpeg libsm6 libxext6 -y \
    && apt-get update \
    && apt-get install -y git curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy requirements
COPY src/train_FashionMNIST.py /app/train_FashionMNIST.py

# Copy the rest of the application code into the container
COPY . .

# Run python script at container startup
CMD ["python", "-u", "train_FashionMNIST.py"]