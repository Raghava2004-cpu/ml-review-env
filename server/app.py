from server import app

def main():
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860)
