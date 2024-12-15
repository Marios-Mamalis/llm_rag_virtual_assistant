from fastapi import FastAPI


app = FastAPI()

@app.get("/health")
def check_status():
    return {"status message": "Server is up."}
