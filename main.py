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
async def train(params: TrainParameters):
    print("Model Training Started")
    print(params)
    return {"message":"train"}

@app.post("/predict")
async def predict(data:RoboAdviserSample):
    print("Predicting")
    print(data)
    return {"message":"predict"}
