FROM python:3.6.3

WORKDIR /usr/src/app

RUN pip install --upgrade pip

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir processed
RUN mkdir checkpoints

COPY src/*.py ./
COPY src/processed ./processed
COPY src/checkpoints ./checkpoints

EXPOSE 5000
CMD [ "python", "telegram_bot.py" ]
