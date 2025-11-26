# Fracture Detection AI (Streamlit)

This repository runs a Streamlit web app that performs fracture detection on X-ray images using a YOLOv8 model and generates human-readable explanations via Google Gemini.

## Quick notes

- Main entrypoint: `app.py`
- Model file: `model.pt` (small, ~6 MB) — included in the repo
- Secrets: `GOOGLE_API_KEY` is required for Gemini. Do NOT commit your `.env` file.

## Local setup

1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. In Streamlit Cloud, create a new app and connect your GitHub repo. Use `app.py` as the main file.
3. In the Streamlit Cloud app settings, add the following secret (do NOT commit this to the repo):
   - `GOOGLE_API_KEY` — your Google Gemini API key
4. Deploy and open the app. Upload an X-ray image to test.

Notes:

- The model file `model.pt` is small (6 MB) and can be stored in the repository. If you later replace it with a large model (>100 MB), host it externally (S3/Hugging Face/etc.) and modify `load_yolo_model()` to download it at runtime.
- If Streamlit Cloud fails to install `torch`/`torchvision` binaries, consider using CPU-only wheels or moving inference to a separate server.

## Environment variables and secrets

- Local development: put `GOOGLE_API_KEY` in a `.env` file (the app loads it via `python-dotenv`).
- Streamlit Cloud: set `GOOGLE_API_KEY` in the Secrets section of the app settings.

## Troubleshooting

- If VS Code shows missing imports, select the workspace virtual environment (`venv`) as the Python interpreter.
- `pyperclip` clipboard functions may not work in headless cloud environments; the app already catches clipboard errors.

## License & Disclaimer

For research and educational purposes only. Not medical advice or diagnosis.
