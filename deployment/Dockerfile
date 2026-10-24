# Starter placeholder: finalize after packaging and API entry points are implemented.
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml requirements-dev.txt ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]"
CMD ["python", "-c", "print('Implement the API entry point before deploying.')"]
