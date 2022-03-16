
from fastapi import FastAPI
from models.model import SentimentModel, SentimentQueryModel

app = FastAPI()
model = SentimentModel()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post('/predict')
def predict(data: SentimentQueryModel):
    data = data.dict()
    polarity, subjectivity = model.get_sentiment(data['text'])
    return { 'polarity': polarity,
                'subjectivity': subjectivity
    }