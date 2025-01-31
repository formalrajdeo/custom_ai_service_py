# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install build-essential, gcc, and cmake to allow building C extensions
RUN apt-get update && \
    apt-get install -y build-essential gcc cmake && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port FastAPI will run on
EXPOSE 8002

# Run FastAPI using Uvicorn (without the environment variable)
CMD ["sh", "-c", "$UVICORN_CMD"]
