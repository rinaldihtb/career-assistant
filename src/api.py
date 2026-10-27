"""FastAPI entry point for the Career Assistant services."""

import uvicorn
from fastapi import FastAPI

from src.routes.routes import router as chat_router

app = FastAPI(title="Career Assistant API", version="0.1.0")
app.include_router(chat_router)

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
