import uvicorn
from fastapi import FastAPI
from routes import router
from core.config import settings

app = FastAPI(
    title="Validator API",
)

app.include_router(router, prefix="/api")

reload = True if settings.ENV == "dev" else False

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=settings.INTERNAL_PORT, reload=reload)