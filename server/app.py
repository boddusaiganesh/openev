import sys
import os
import uvicorn

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from server import app

def main():
    uvicorn.run(app, host='0.0.0.0', port=7860)

if __name__ == '__main__':
    main()
