import uvicorn

from Linear import Linear
from main import app


if __name__ == '__main__':
    uvicorn.run(app)