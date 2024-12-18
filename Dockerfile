FROM python:3.10.16-alpine3.21@sha256:748b5868188a58e05375eb70972cbdb338bae30c6e613a847910315e3d20afc4

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir --force-reinstall -r requirements.txt

RUN mkdir /logs

COPY . /app/

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
