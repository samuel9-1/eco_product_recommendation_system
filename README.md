# Eco Product Recommendation System

A Flask web app using Pandas and scikit-learn TF-IDF similarity plus transparent ranking heuristics.

## Features
- Search product descriptions with keywords
- Filter by category, budget and minimum rating
- Rank by low plastic, recycled materials, durability, low carbon or vegan preferences
- Responsive interface and `/health` status endpoint

## Dataset disclaimer
`data/eco_products.csv` is a **synthetic demo dataset**. Prices, ratings and carbon-footprint values are illustrative and not verified claims. Replace with a sourced dataset before real-world use. Match percentage is a ranking indicator, not a certified sustainability score.

## Run on Windows / VS Code
1. Install Python 3.10+ and open this folder in VS Code.
2. Run `py -m venv .venv`
3. PowerShell: `\.venv\Scripts\Activate.ps1`
   - If blocked, use Command Prompt: `.venv\Scripts\activate.bat`
4. `python -m pip install --upgrade pip`
5. `pip install -r requirements.txt`
6. `python app.py`
7. Open `http://127.0.0.1:5000`

Run commands from the folder containing `app.py` and `requirements.txt`.

## Production deployment
For a Python host, add `gunicorn` to requirements and use `gunicorn app:app` as the start command. `debug=True` is for local development only.
