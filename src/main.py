from fastapi import FastAPI
from src.routes import v1_router


app = FastAPI()
app.include_router(v1_router)

