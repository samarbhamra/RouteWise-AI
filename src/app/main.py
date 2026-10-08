"""ASGI / CLI entrypoint: python -m src.app.main"""
from src.app.api import app, main

__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
