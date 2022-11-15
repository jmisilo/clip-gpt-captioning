# for tests on Linux machines
FROM python:3.9.13

# Set the working directory to /app
WORKDIR /app

RUN python -m venv venv

RUN . venv/bin/activate

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY /src /app/src

CMD ["python", "-u", "src/training.py"]