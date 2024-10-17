### Pip stage ###
FROM python:3.10-slim as compiler
ENV PYTHONUNBUFFERED 1

ADD . .

RUN python -m venv /venv

ENV PATH="/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

### Runner stage ###
FROM python:3.10-slim as runner

COPY --from=compiler /venv /venv
COPY src /src

ENV PATH="/venv/bin:$PATH"

CMD fastapi run src/deploy/fastapp.py --port 7999