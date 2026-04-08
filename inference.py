from fastapi import FastAPI
import uvicorn

from inference import run_inference

app = FastAPI(
    title="OpenEnv Log Triage Agent"
)

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset():
    """
    Endpoint required by validator
    """
    result = run_inference()
    return {
        "status": "reset complete",
        "result": result
    }


def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860
    )
