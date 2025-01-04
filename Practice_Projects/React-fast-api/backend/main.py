import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import ollama
from ollama import Client

host = 'https://select-indirectly-jennet.ngrok-free.app'
# host = 'http://localhost:11434'

model_en = 'llama3.2'
model_hi2 = 'SL-Lexicons/llama3-hindi-8b-q5_km.gguf:latest'
model_hi = 'ayansh03/hindi-gemma:q8_0'
model_te = 'rohithbojja/llama3-telugu:latest'
model_ta = 'conceptsintamil/tamil-llama-7b-instruct-v0.2:latest'

# try:
#   ollama.chat(model)
# except ollama.ResponseError as e:
#   print('Error:', e.error)
#   if e.status_code == 404:
#     ollama.pull(model)

client = Client(
  host=host#,
#   headers={'x-some-header': 'some-value'}
)

def answer_en(msg):
    response = client.chat(model=model_en, messages=[
    {
        'role': 'user',
        'content': msg,
    },
    ])
    ai = f"ChatBot : {response.message.content}"
    return ai

def answer(msg):
    lang="en"
    if lang=="en":
        answer_en(msg)

class Fruit(BaseModel):
    name: str

class Fruits(BaseModel):
    fruits: List[Fruit]
    
app = FastAPI(debug=True)

origins = [
    "http://localhost:5173",
    # Add more origins here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory_db = {"fruits": []}

@app.get("/fruits", response_model=Fruits)
def get_fruits():
    return Fruits(fruits=memory_db["fruits"])

@app.post("/fruits")
def add_fruit(fruit: Fruit):
    fruit.name = str(answer(str(fruit.name)))
    memory_db["fruits"].append(fruit)
    return fruit
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4173)