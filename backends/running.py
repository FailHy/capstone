import uvicorn

if __name__ == "__main__":
    # Runs the app on http://0.0.0.0:8000
    # reload=True is great for development (auto-restarts on save)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)