"""AWS Lambda adapter for the FastAPI app."""

from mangum import Mangum

from src.api import app

handler = Mangum(app, lifespan="off")
