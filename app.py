from flask import Flask, render_template, request
from recommender import EcoProductRecommender

app = Flask(__name__)
recommender = EcoProductRecommender("data/eco_products.csv")
CATEGORIES = ["All", "Home", "Personal Care", "Fashion", "Kitchen", "Electronics", "Outdoor"]
BUDGETS = ["Any", "Under ₹500", "₹500–₹1,500", "₹1,500–₹3,000", "Over ₹3,000"]

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations, error = [], None
    form = {"category":"All", "budget":"Any", "priorities":["Low Plastic"], "query":"", "min_rating":"3.5"}
    if request.method == "POST":
        form = {
            "category": request.form.get("category", "All"),
            "budget": request.form.get("budget", "Any"),
            "priorities": request.form.getlist("priorities"),
            "query": request.form.get("query", "").strip(),
            "min_rating": request.form.get("min_rating", "3.5"),
        }
        try:
            recommendations = recommender.recommend(
                category=form["category"], budget=form["budget"],
                priorities=form["priorities"], query=form["query"],
                min_rating=float(form["min_rating"]), top_n=8)
            if not recommendations:
                error = "No products matched every filter. Try a wider budget, another category, or a lower rating threshold."
        except ValueError as exc:
            error = str(exc)
    return render_template("index.html", recommendations=recommendations, error=error,
        form=form, categories=CATEGORIES, budgets=BUDGETS, total_products=recommender.product_count)

@app.route("/health")
def health():
    return {"status":"ok", "products_loaded":recommender.product_count}

if __name__ == "__main__":
    app.run(debug=True)
