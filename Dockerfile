# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8

WORKDIR /code

# Install pip requirements
COPY ./requirements.txt /code/requirements.txt


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get -y install gcc mono-mcs && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade -r  requirements.txt

COPY ./app /code/app


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]