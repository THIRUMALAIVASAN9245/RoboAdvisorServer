from fastapi import FastAPI
from model_trainer import ModelTrainer
from train_parameters import TrainParameters
from robo_adviser_sample import RoboAdviserSample

app = FastAPI(title="REST API using FastAPI Async EndPoints")

app.model = ModelTrainer('svm')

@app.get("/sample")
def sample():
    return {"message":"Hello Thirumalai"}

@app.post("/train")
async def train(params: TrainParameters):
    print("Model Training Started")
    app.model =  ModelTrainer(params.model.lower(), params.testsize)
    accuracy = app.model.train(params.path)
    return accuracy

@app.post("/predict")
async def predict(data:RoboAdviserSample):
    print("Predicting")
    spicies_map = {0: 'Basic', 1: 'Standard', 2: 'Premium'}
    profileType = app.model.predict(data)
    return spicies_map[profileType[0]]
