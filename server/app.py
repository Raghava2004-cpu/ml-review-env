import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

def main():
    uvicorn.run("server:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
