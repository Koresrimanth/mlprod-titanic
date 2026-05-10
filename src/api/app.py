from fastapi import FastAPI

from src.api.routes import router


app = FastAPI(
    title="Titanic Survival Prediction API"
)


app.include_router(router)
