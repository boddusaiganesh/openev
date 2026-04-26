"""
server/app.py — Entry point shim for the LexArena server.
Routes to lexarena_server.py which is the unified 6-tier FastAPI app.
"""
import sys
import os
import uvicorn

# Ensure the project root is on sys.path
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lexarena_server import app  # noqa: E402


def main():
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
