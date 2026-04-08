from fastapi import FastAPI
import uvicorn
import json

# Import your inference logic
# Make sure inference.py is in repo root
from inference import run_inference

app = FastAPI(
    title="OpenEnv Log Triage Agent",
    description="RL Agent Server for Log Triage",
    version="1.0"
)

# Health check endpoint (very important)
@app.get("/")
def root():
    return {"status": "ok"}


# Inference endpoint
@app.post("/infer")
def infer(payload: dict):
    """
    Receives log data and returns prediction.
    """
    try:
        result = run_inference(payload)
        return {"result": result}

    except Exception as e:
        return {
            "error": str(e)
        }


# Required entry point for deployment
def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860
    )
