from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/")
async def read_root():
    return {
        "message": "Halo API is UP!"
    }

# Run
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

