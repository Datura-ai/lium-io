FROM python:3.11-slim

RUN pip install -U pdm
ENV PDM_CHECK_UPDATE=false

COPY ../../datura /datura

COPY pyproject.toml pdm.lock README.md /root/app/

WORKDIR /root/app

RUN pdm install --check --prod --no-editable

ENV PATH="/root/app/.venv/bin:$PATH"

COPY . .

CMD ["bash", "run.sh"]