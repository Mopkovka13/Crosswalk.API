from fastapi import FastAPI, UploadFile
from methods import get_result


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/get_crosswalk_class")
async def get_crosswalk_class(file: UploadFile):
    result = -1

    image_file = file.file

    result = get_result(image_file)

    return {"crosswalk_class": result}