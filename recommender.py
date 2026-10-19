from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class EcoProductRecommender:
    """Content-based recommender; ranking is a demo heuristic, not a verified eco-certification."""
    def __init__(self, csv_path):
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
        self.products = pd.read_csv(path).fillna("")
        required = {"product_id","name","category","price","rating","description","materials",
                    "eco_features","packaging","durability","carbon_footprint_kg"}
        missing = required - set(self.products.columns)
        if missing:
            raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
        for col in ["price","rating","carbon_footprint_kg"]:
            self.products[col] = pd.to_numeric(self.products[col], errors="coerce").fillna(0)
        text = (self.products["name"].astype(str) + " " + self.products["category"].astype(str) + " " +
                self.products["description"].astype(str) + " " + self.products["materials"].astype(str) + " " +
                self.products["eco_features"].astype(str) + " " + self.products["packaging"].astype(str))
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.product_matrix = self.vectorizer.fit_transform(text)
        self.product_count = len(self.products)

    def _budget_filter(self, prices, budget):
        if budget == "Under ₹500": return prices < 500
        if budget == "₹500–₹1,500": return (prices >= 500) & (prices <= 1500)
        if budget == "₹1,500–₹3,000": return (prices > 1500) & (prices <= 3000)
        if budget == "Over ₹3,000": return prices > 3000
        return pd.Series(True, index=prices.index)

    def recommend(self, category="All", budget="Any", priorities=None, query="", min_rating=3.5, top_n=8):
        priorities = priorities or []
        candidates = self.products.copy()
        mask = pd.Series(True, index=candidates.index)
        if category and category != "All":
            mask &= candidates["category"].str.casefold() == category.casefold()
        mask &= self._budget_filter(candidates["price"], budget)
        mask &= candidates["rating"] >= float(min_rating)
        candidates = candidates.loc[mask].copy()
        if candidates.empty: return []
        if query:
            query_vec = self.vectorizer.transform([query])
            query_scores = cosine_similarity(query_vec, self.product_matrix).ravel()
            candidates["query_score"] = query_scores[candidates.index]
        else:
            candidates["query_score"] = 0.0
        features = candidates["eco_features"].astype(str).str.lower()
        materials = candidates["materials"].astype(str).str.lower()
        packaging = candidates["packaging"].astype(str).str.lower()
        score = candidates["rating"] / 5.0 * 0.20
        score += (1.0 / (1.0 + candidates["carbon_footprint_kg"])) * 0.25
        score += candidates["durability"].astype(str).str.lower().isin(["high","very high"]).astype(float) * 0.15
        score += candidates["query_score"] * 0.20
        for priority in priorities:
            p = priority.casefold()
            if p == "low plastic":
                score += packaging.str.contains("plastic-free|recycled paper|minimal", regex=True).astype(float) * 0.10
                score += features.str.contains("plastic-free|refillable|reusable", regex=True).astype(float) * 0.05
            elif p == "recycled materials":
                score += materials.str.contains("recycled|reclaimed|upcycled", regex=True).astype(float) * 0.15
            elif p == "long lasting":
                score += candidates["durability"].astype(str).str.lower().isin(["high","very high"]).astype(float) * 0.15
            elif p == "low carbon":
                score += (1.0 / (1.0 + candidates["carbon_footprint_kg"])) * 0.15
            elif p == "vegan":
                score += features.str.contains("vegan|plant-based|cruelty-free", regex=True).astype(float) * 0.15
        candidates["match_score"] = score.clip(0, 1)
        candidates = candidates.sort_values(["match_score","rating","price"], ascending=[False,False,True]).head(top_n)
        return [{
            "product_id":str(r["product_id"]), "name":str(r["name"]), "category":str(r["category"]),
            "price":float(r["price"]), "rating":float(r["rating"]), "description":str(r["description"]),
            "materials":str(r["materials"]), "eco_features":str(r["eco_features"]), "packaging":str(r["packaging"]),
            "durability":str(r["durability"]), "carbon_footprint_kg":float(r["carbon_footprint_kg"]),
            "match_percent":round(float(r["match_score"])*100)
        } for _, r in candidates.iterrows()]
