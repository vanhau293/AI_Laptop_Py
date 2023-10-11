import sys
sys.path.append('./model_laptop_text/model_laptop_text')
sys.path.append('./svm_laptop/svm_laptop')
from predict_text_laptop import predict_demand
from predict_model import predict_laptop
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
#Make POST API
app = FastAPI()
class Laptop(BaseModel):
    price: str
    ram: str
    hard_drive: str
    weight: str
    brand_rank: str
    cpu_rank: str
    gpu_rank: str
    inch: str

@app.post('/api/laptop')
def process_laptop(laptop: Laptop):
    # Access the attributes of the payload
    price= laptop.price
    ram= laptop.ram 
    hard_drive= laptop.hard_drive
    weight= laptop.weight
    brand_rank= laptop.brand_rank 
    cpu_rank= laptop.cpu_rank 
    gpu_rank= laptop.gpu_rank 
    inch= laptop.inch 

    # Process the payload
    # ...
    input_laptop = [price, ram, hard_drive, weight, brand_rank, cpu_rank, gpu_rank, inch]
    print(input_laptop)
    temp = predict_laptop(input_laptop)
    return str(temp)

class Demand(BaseModel):
    info: str

@app.post('/api/demand')
def process_demand(demand: Demand):
    info = demand.info
    print(info)
    temp = predict_demand(info)
    return str(temp)
if __name__ == '__main__':
    uvicorn.run("main:app", port=5000, host='ailaptoppy-production.up.railway.app' reload=True, access_log=False)
