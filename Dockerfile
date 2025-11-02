# Dockerfile (for Hugging Face Spaces)
FROM python:3.10-slim

WORKDIR /app

# copy files
COPY . /app

# pre-install build deps if needed (e.g. for scikit-learn)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# install python deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# expose port
EXPOSE 5000

# start gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "1", "--threads", "2"]
