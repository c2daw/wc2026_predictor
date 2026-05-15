"""Run this locally to regenerate model_cache.json after new match results."""
import json
import numpy as np
from datetime import datetime
from model import load_data, fit_model

MIN_DATE = "2014-01-01"

print("Fetching data and fitting model...")
df = load_data(min_date=MIN_DATE)
ratings, home_adv, rho, valid_teams, params = fit_model(df)

cache = {
    "generated_at": datetime.utcnow().isoformat(),
    "home_adv": float(home_adv),
    "rho": float(rho),
    "valid_teams": valid_teams,
    "params": params.tolist(),
    "ratings": ratings.reset_index().rename(columns={"index": "rank"}).to_dict(orient="records"),
}

with open("model_cache.json", "w") as f:
    json.dump(cache, f)

print(f"Done — {len(valid_teams)} teams. Saved to model_cache.json")
