"""ASGI entry point: uvicorn eurosat_mlops.api.main:app"""

from eurosat_mlops.api.app import create_app

app = create_app()
