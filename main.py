from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model_trainer import ModelTrainer
from train_parameters import TrainParameters
from robo_adviser_sample import RoboAdviserSample

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.model = ModelTrainer('svm')

@app.get("/")
def home():
    return {"message":"Hello Thirumalai"}

@app.get("/sample")
def sample():
    return {"message":"Hello Thirumalai"}

@app.post("/train")
def train(params):
    print("Model Training Started")
    return {"message":"train"}

@app.post("/predict")
def predict(data):
    print("Predicting")
    return {"message":"predict"}
