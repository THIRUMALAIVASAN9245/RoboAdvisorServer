from fastapi import FastAPI
from model_trainer import ModelTrainer
from train_parameters import TrainParameters
from robo_adviser_sample import RoboAdviserSample

app = FastAPI(title="REST API using FastAPI Async EndPoints")

app.model = ModelTrainer('svm')

@app.get("/")
def home():
    return {"message":"Hello Thirumalai"}

@app.get("/sample")
def sample():
    return {"message":"Hello Thirumalai"}

@app.post("/train")
def train(params:TrainParameters):
    print("Model Training Started")
    return {"message":"train"}

@app.post("/predict")
def predict(data:RoboAdviserSample):
    print("Predicting")
    return {"message":"predict"}
