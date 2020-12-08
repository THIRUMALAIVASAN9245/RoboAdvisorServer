from fastapi import FastAPI

app = FastAPI()

#domain where this api is hosted for example : localhost:5000/docs to see swagger documentation automagically generated.


@app.get("/")
def home():
    return {"message":"Hello Thirumalai"}

@app.get("/sample")
def home():
    return {"message":"Hello Thirumalai"}

@app.post("/train")
def train(params: TrainParameters):
    return {"message":"train"}

@app.post("/predict")
def predict(data:RoboAdviserSample):
    return {"message":"predict"}
