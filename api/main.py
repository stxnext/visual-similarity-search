import os
from typing import Callable

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.timing import add_timing_middleware
from loguru import logger

from api.routes import router
from metrics.core import MetricClient

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
]
APP_SETTINGS = {
    "title": "Metrics Learning Api",
    "description": "Machine learning metrics project",
    "docs_url": "/",
}


def start_app_handler(app: FastAPI) -> Callable:
    """
    Create metric client with loaded models and assign it into app memory/state,
    so we won't have cold start on any endpoint
    """

    def startup() -> None:
        logger.info("Running app start handler.")
        app.state.metric_client = MetricClient()

    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    """Flush app state"""

    def shutdown() -> None:
        logger.info("Running app shutdown handler.")
        del app.state.metric_client

    return shutdown


def get_app() -> FastAPI:
    """Create fastapi app with middlewares and handlers"""
    app = FastAPI(**APP_SETTINGS)
    app.include_router(router, prefix="/api")
    app.add_event_handler("startup", start_app_handler(app))
    app.add_event_handler("shutdown", stop_app_handler(app))
    if os.getenv("TYPE") != "PROD":
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    add_timing_middleware(app, record=logger.info, prefix="app")
    return app


app = get_app()


if __name__ == "__main__":
    uvicorn.run(
        app="main:app",
        host=os.getenv("API_HOST", "localhost"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
    )
