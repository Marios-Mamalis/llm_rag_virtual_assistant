import typing
from fastapi import FastAPI


app = FastAPI()


@app.get("/health")
def check_status() -> typing.Dict[str, str]:
    return {"status message": "Server is up."}

