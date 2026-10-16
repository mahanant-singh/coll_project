from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

le_game = joblib.load("game_encoder.pkl")
le_level = joblib.load("level_encoder.pkl")
model = joblib.load("xgb_model.pkl")


class Features(BaseModel):
    age: int
    game_type: str
    score: float
    accuracy: float
    response_time_sec: float
    mistakes: int
    games_played: int


@app.get("/")
def home():
    return {"message": "Hello guys!"}


@app.post("/predict")
def predict(features: Features):
    data = features.model_dump()
    df = pd.DataFrame([data])

    prediction = model.predict(df)

    game = le_game.inverse_transform(
        [int(prediction[0][0])]
    )[0]

    level = le_level.inverse_transform(
        [int(prediction[0][1])]
    )[0]

    return {
        "recommended_game": str(game),
        "recommended_level": str(level)
    }