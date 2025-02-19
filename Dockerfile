FROM python:3.9

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .

# install python dependencies
RUN pip install --upgrade pip
#RUN pip install --no-cache-dir -r requirements.txt
RUN while read requirement; do pip install --no-cache-dir $requirement; done < requirements.txt

COPY env.sample .env

COPY . .

# running migrations
RUN python manage.py makemigrations
RUN python manage.py migrate
RUN python manage.py generate-api

# gunicorn
CMD ["gunicorn", "--config", "gunicorn-cfg.py", "core.wsgi"]

