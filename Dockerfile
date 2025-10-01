# Use the official Python image as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg

# Copy the requirements file into the container
COPY requirements.txt .

# Install torch, torchaudio, and torchvision from the specified index URL
RUN pip install --no-cache-dir --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126/ torch torchaudio torchvision

# Install the other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code into the container
COPY ./backend /app/backend

# Expose the port the app runs on
EXPOSE 8600

# Command to run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8600"]
