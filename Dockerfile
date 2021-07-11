FROM python:3.7.10
ENV DEBIAN_FRONTEND=noninteractive 

WORKDIR /app
COPY . .

RUN apt update
RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install -r requirements.txt

CMD ["python3.7", "app.py"]