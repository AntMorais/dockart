FROM python:3.8

RUN pip install fastapi uvicorn spacy pydantic torch torchvision pandas numpy matplotlib

COPY . /api/api

ENV PYTHONPATH=/api
WORKDIR /api

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["api.main:app", "--host", "0.0.0.0"]