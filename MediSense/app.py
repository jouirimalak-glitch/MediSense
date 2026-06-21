"""
PharmaPred — API REST de prédiction de demande de médicaments
Flask · 10 endpoints · JSON · CORS activé
Lancez : python app.py
"""
import os
import json
import sys
import numpy as np
import pandas as pd
from flask import Flask, request, Response

# ── Chemins absolus (fonctionne partout : terminal, double-clic, IDE) ──
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Import du modèle ─────────────────────────────────────────
sys.path.insert(0, BASE_DIR)
from train_model import PharmacyDemandPredictor

app = Flask(__name__)

# ── Chargement au démarrage ──────────────────────────────────
print("Chargement des données...", flush=True)
try:
    sales_df = pd.read_csv(os.path.join(DATA_DIR, "sales_history.csv"))
    stock_df = pd.read_csv(os.path.join(DATA_DIR, "stock_history.csv"))
    med_df   = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))
except FileNotFoundError as e:
    print(f"❌ Fichier manquant : {e}")
    print("   Lancez d'abord : python generate_data.py")
    sys.exit(1)

MEDS = med_df.set_index("id").to_dict("index")

print("Chargement des modèles...", flush=True)
pkl_path = os.path.join(MODELS_DIR, "demand_predictor.pkl")
if not os.path.exists(pkl_path):
    print(f"❌ Modèle introuvable : {pkl_path}")
    print("   Lancez d'abord : python train_model.py")
    sys.exit(1)

predictor = PharmacyDemandPredictor.load(pkl_path)


# ── Helper : réponse JSON + CORS ────────────────────────────
def jsn(data, status=200):
    r = Response(
        json.dumps(data, ensure_ascii=False, default=str),
        status=status, mimetype="application/json"
    )
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return r

@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        r = Response()
        r.headers["Access-Control-Allow-Origin"]  = "*"
        r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return r


# ── Endpoint 1 — Documentation ───────────────────────────────
@app.route("/", methods=["GET"])
def root():
    avg_acc = float(np.mean([m["accuracy"] for m in predictor.metrics.values()]))
    return jsn({
        "api":     "PharmaPred — Medication Demand Prediction API",
        "version": "1.0.0",
        "endpoints": [
            {"method":"GET",  "path":"/",                         "description":"Documentation"},
            {"method":"GET",  "path":"/api/medications",          "description":"Liste tous les médicaments"},
            {"method":"GET",  "path":"/api/medications/<id>",     "description":"Détails d'un médicament"},
            {"method":"GET",  "path":"/api/predict/<id>",         "description":"Prédictions (param: horizon)"},
            {"method":"GET",  "path":"/api/recommend/<id>",       "description":"Recommandation réapprovisionnement"},
            {"method":"GET",  "path":"/api/recommend/all",        "description":"Toutes les recommandations"},
            {"method":"GET",  "path":"/api/analytics/sales",      "description":"Analytique des ventes"},
            {"method":"GET",  "path":"/api/analytics/top-movers", "description":"Top médicaments"},
            {"method":"GET",  "path":"/api/model/metrics",        "description":"Métriques ML"},
            {"method":"POST", "path":"/api/predict/custom",       "description":"Prédiction personnalisée"},
        ],
        "model": {
            "algorithm":    "Random Forest Regressor",
            "medications":  len(predictor.models),
            "avg_accuracy": f"{avg_acc:.1f}%",
            "data_records": len(sales_df),
        },
    })


# ── Endpoint 2 — Liste médicaments ──────────────────────────
@app.route("/api/medications", methods=["GET"])
def get_medications():
    result = []
    for mid, info in MEDS.items():
        ms    = stock_df[stock_df["medication_id"] == mid]
        stock = int(ms.sort_values("period").iloc[-1]["closing_stock"]) if len(ms) else 0
        result.append({
            "id":             mid,
            "name":           info["name"],
            "category":       info["category"],
            "unit_cost":      info["unit_cost"],
            "reorder_point":  info["reorder_point"],
            "current_stock":  stock,
            "has_model":      mid in predictor.models,
            "model_accuracy": predictor.metrics.get(mid, {}).get("accuracy", 0),
        })
    return jsn({"total": len(result), "medications": result})


# ── Endpoint 3 — Détail médicament ──────────────────────────
@app.route("/api/medications/<med_id>", methods=["GET"])
def get_medication(med_id):
    med_id = med_id.upper()
    if med_id not in MEDS:
        return jsn({"error": f"Médicament {med_id} introuvable"}, 404)
    info = MEDS[med_id]
    ms   = sales_df[sales_df["medication_id"] == med_id].copy()
    ms["date"] = pd.to_datetime(ms["date"])
    monthly = ms.groupby(ms["date"].dt.to_period("M")).agg(
        {"quantity_sold": "sum", "revenue": "sum"}
    ).reset_index()
    st    = stock_df[stock_df["medication_id"] == med_id]
    stock = int(st.sort_values("period").iloc[-1]["closing_stock"]) if len(st) else 0
    return jsn({
        "id": med_id, **info,
        "current_stock":    stock,
        "total_records":    len(ms),
        "total_revenue":    round(float(ms["revenue"].sum()), 2),
        "avg_daily_demand": round(float(ms["quantity_sold"].mean()), 1),
        "monthly_history":  [
            {"period": str(r["date"]), "qty": int(r["quantity_sold"]),
             "revenue": round(float(r["revenue"]), 2)}
            for _, r in monthly.tail(24).iterrows()
        ],
        "model_metrics": predictor.metrics.get(med_id, {}),
    })


# ── Endpoint 4 — Prédictions ────────────────────────────────
@app.route("/api/predict/<med_id>", methods=["GET"])
def predict(med_id):
    med_id  = med_id.upper()
    horizon = min(int(request.args.get("horizon", 30)), 90)
    if med_id not in predictor.models:
        return jsn({"error": f"Pas de modèle pour {med_id}"}, 404)
    preds = predictor.predict_demand(sales_df, med_id, horizon)
    total = sum(p["predicted_demand"] for p in preds)
    return jsn({
        "medication_id":   med_id,
        "medication_name": MEDS.get(med_id, {}).get("name", med_id),
        "horizon_days":    horizon,
        "predictions":     preds,
        "summary": {
            "total_predicted_demand": total,
            "avg_daily_demand":       round(total / horizon, 1),
            "max_daily":              max(p["predicted_demand"] for p in preds),
            "min_daily":              min(p["predicted_demand"] for p in preds),
        },
        "model_metrics": predictor.metrics.get(med_id, {}),
    })


# ── Endpoint 5 — Recommandation ─────────────────────────────
@app.route("/api/recommend/<med_id>", methods=["GET"])
def recommend(med_id):
    if med_id.upper() == "ALL":
        return recommend_all()
    med_id  = med_id.upper()
    horizon = min(int(request.args.get("horizon", 30)), 90)
    safety  = float(request.args.get("safety_factor", 1.2))
    if med_id not in predictor.models:
        return jsn({"error": f"Pas de modèle pour {med_id}"}, 404)
    rec  = predictor.recommend_restock(sales_df, stock_df, med_id, horizon, safety)
    info = MEDS.get(med_id, {})
    rec["medication_name"] = info.get("name", med_id)
    rec["category"]        = info.get("category", "")
    rec["estimated_cost"]  = round(rec["recommended_order_quantity"] * info.get("unit_cost", 0), 2)
    return jsn(rec)


# ── Endpoint 6 — Toutes recommandations ─────────────────────
@app.route("/api/recommend/all", methods=["GET"])
def recommend_all():
    horizon    = min(int(request.args.get("horizon", 30)), 90)
    safety     = float(request.args.get("safety_factor", 1.2))
    recs       = []
    total_cost = 0.0
    for mid in predictor.models:
        rec  = predictor.recommend_restock(sales_df, stock_df, mid, horizon, safety)
        info = MEDS.get(mid, {})
        rec["medication_name"] = info.get("name", mid)
        rec["category"]        = info.get("category", "")
        rec["estimated_cost"]  = round(rec["recommended_order_quantity"] * info.get("unit_cost", 0), 2)
        total_cost += rec["estimated_cost"]
        recs.append(rec)
    recs.sort(key=lambda x: (-x["urgency_level"], -x["recommended_order_quantity"]))
    by = {3: 0, 2: 0, 1: 0, 0: 0}
    for r in recs:
        by[r["urgency_level"]] += 1
    return jsn({
        "total_medications": len(recs),
        "summary": {
            "critical": by[3], "urgent": by[2],
            "moderate": by[1], "ok":     by[0],
            "total_estimated_restock_cost": round(total_cost, 2),
        },
        "recommendations": recs,
    })


# ── Endpoint 7 — Analytique ventes ──────────────────────────
@app.route("/api/analytics/sales", methods=["GET"])
def analytics():
    df = sales_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.groupby(df["date"].dt.to_period("M")).agg(
        {"quantity_sold": "sum", "revenue": "sum"}
    ).reset_index()
    by_cat = df.groupby("category").agg(
        {"quantity_sold": "sum", "revenue": "sum"}
    ).reset_index()
    by_year = df.groupby(df["date"].dt.year).agg(
        {"quantity_sold": "sum", "revenue": "sum"}
    ).reset_index()
    return jsn({
        "total_records": len(df),
        "date_range":    {"start": str(df["date"].min().date()), "end": str(df["date"].max().date())},
        "totals":        {"quantity_sold": int(df["quantity_sold"].sum()), "revenue": round(float(df["revenue"].sum()), 2)},
        "monthly_sales": [{"period": str(r["date"]), "qty": int(r["quantity_sold"]), "revenue": round(float(r["revenue"]), 2)} for _, r in monthly.iterrows()],
        "by_category":   [{"category": r["category"], "qty": int(r["quantity_sold"]), "revenue": round(float(r["revenue"]), 2)} for _, r in by_cat.sort_values("revenue", ascending=False).iterrows()],
        "by_year":       [{"year": int(r["date"]), "qty": int(r["quantity_sold"]), "revenue": round(float(r["revenue"]), 2)} for _, r in by_year.iterrows()],
    })


# ── Endpoint 8 — Top movers ─────────────────────────────────
@app.route("/api/analytics/top-movers", methods=["GET"])
def top_movers():
    tq = sales_df.groupby(["medication_id", "medication_name"])["quantity_sold"].sum().reset_index().sort_values("quantity_sold", ascending=False).head(10)
    tr = sales_df.groupby(["medication_id", "medication_name"])["revenue"].sum().reset_index().sort_values("revenue", ascending=False).head(10)
    return jsn({
        "top_by_quantity": [{"rank": i+1, "id": r["medication_id"], "name": r["medication_name"], "total_sold": int(r["quantity_sold"])} for i, (_, r) in enumerate(tq.iterrows())],
        "top_by_revenue":  [{"rank": i+1, "id": r["medication_id"], "name": r["medication_name"], "total_revenue": round(float(r["revenue"]), 2)} for i, (_, r) in enumerate(tr.iterrows())],
    })


# ── Endpoint 9 — Métriques ML ───────────────────────────────
@app.route("/api/model/metrics", methods=["GET"])
def model_metrics():
    rows    = [{"medication_id": mid, "medication_name": MEDS.get(mid, {}).get("name", mid), **m} for mid, m in predictor.metrics.items()]
    avg_r2  = float(np.mean([m["r2"]       for m in predictor.metrics.values()]))
    avg_mae = float(np.mean([m["mae"]      for m in predictor.metrics.values()]))
    avg_acc = float(np.mean([m["accuracy"] for m in predictor.metrics.values()]))
    return jsn({
        "algorithm": "Random Forest Regressor",
        "features":  PharmacyDemandPredictor.FEATURE_COLS,
        "global_performance": {
            "avg_r2":       round(avg_r2, 3),
            "avg_mae":      round(avg_mae, 2),
            "avg_accuracy": round(avg_acc, 1),
        },
        "per_medication": rows,
    })


# ── Endpoint 10 — Prédiction personnalisée ──────────────────
@app.route("/api/predict/custom", methods=["POST"])
def predict_custom():
    body    = request.get_json()
    if not body:
        return jsn({"error": "Body JSON requis"}, 400)
    med_id  = body.get("medication_id", "").upper()
    horizon = min(int(body.get("horizon", 30)), 90)
    safety  = float(body.get("safety_factor", 1.2))
    if not med_id:
        return jsn({"error": "medication_id requis"}, 400)
    if med_id not in predictor.models:
        return jsn({"error": f"Pas de modèle pour {med_id}"}, 404)
    preds = predictor.predict_demand(sales_df, med_id, horizon)
    rec   = predictor.recommend_restock(sales_df, stock_df, med_id, horizon, safety)
    info  = MEDS.get(med_id, {})
    return jsn({
        "medication_id":   med_id,
        "medication_name": info.get("name", med_id),
        "parameters":      {"horizon": horizon, "safety_factor": safety},
        "predictions":     preds,
        "restocking_recommendation": rec,
        "estimated_cost":  round(rec["recommended_order_quantity"] * info.get("unit_cost", 0), 2),
    })


# ── Lancement ────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  PharmaPred API — Démarrage")
    print("=" * 55)
    print(f"  Dossier    : {BASE_DIR}")
    print(f"  Médicaments: {len(MEDS)}")
    print(f"  Modèles    : {len(predictor.models)}")
    print(f"  Ventes     : {len(sales_df):,} enregistrements")
    print(f"  URL        : http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
