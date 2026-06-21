"""
PharmaPred — Modèle ML de prédiction de demande
Random Forest Regressor · 30 modèles · features temporelles + lags
Lancez : python train_model.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings("ignore")


class PharmacyDemandPredictor:
    """Un modèle Random Forest par médicament."""

    FEATURE_COLS = [
        "year", "month", "day_of_week", "day_of_month",
        "week_of_year", "quarter", "is_weekend",
        "month_sin", "month_cos", "dow_sin", "dow_cos",
        "lag_7", "lag_14", "lag_30", "rolling_7", "rolling_30",
    ]

    def __init__(self):
        self.models  = {}   # {med_id: RandomForestRegressor}
        self.metrics = {}   # {med_id: {r2, mae, rmse, accuracy}}

    # ── Privé : construction des features ───────────────────
    def _add_date_features(self, df):
        df = df.copy()
        df["date"]         = pd.to_datetime(df["date"])
        df["year"]         = df["date"].dt.year
        df["month"]        = df["date"].dt.month
        df["day_of_week"]  = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["quarter"]      = df["date"].dt.quarter
        df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
        df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)
        df["dow_sin"]      = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"]      = np.cos(2 * np.pi * df["day_of_week"] / 7)
        return df

    def _add_lags(self, med_df):
        med_df = med_df.sort_values("date").copy()
        for lag in [7, 14, 30]:
            med_df[f"lag_{lag}"] = med_df["quantity_sold"].shift(lag)
        med_df["rolling_7"]  = med_df["quantity_sold"].shift(1).rolling(7).mean()
        med_df["rolling_30"] = med_df["quantity_sold"].shift(1).rolling(30).mean()
        return med_df.dropna()

    # ── Public : entraînement ────────────────────────────────
    def train(self, sales_df):
        df  = self._add_date_features(sales_df)
        ids = df["medication_id"].unique()

        for med_id in ids:
            med_data = self._add_lags(df[df["medication_id"] == med_id])
            if len(med_data) < 60:
                continue

            X = med_data[self.FEATURE_COLS]
            y = med_data["quantity_sold"]

            split   = int(len(med_data) * 0.8)
            X_tr, X_te = X.iloc[:split], X.iloc[split:]
            y_tr, y_te = y.iloc[:split], y.iloc[split:]

            model = RandomForestRegressor(
                n_estimators=100, max_depth=10,
                min_samples_split=5, random_state=42, n_jobs=-1
            )
            model.fit(X_tr, y_tr)

            y_pred = np.maximum(0, model.predict(X_te))
            mae    = mean_absolute_error(y_te, y_pred)
            rmse   = float(np.sqrt(mean_squared_error(y_te, y_pred)))
            r2     = float(r2_score(y_te, y_pred))
            acc    = round(max(0.0, 1 - mae / float(y_te.mean())) * 100, 1)

            self.models[med_id]  = model
            self.metrics[med_id] = {
                "mae":      round(float(mae), 2),
                "rmse":     round(rmse, 2),
                "r2":       round(r2, 3),
                "accuracy": acc,
            }
        return self

    # ── Public : prédiction journalière ──────────────────────
    def predict_demand(self, sales_df, med_id, horizon=30):
        if med_id not in self.models:
            raise ValueError(f"Pas de modèle pour {med_id}")

        df       = self._add_date_features(sales_df)
        med_data = self._add_lags(df[df["medication_id"] == med_id])
        last_date = pd.to_datetime(med_data.iloc[-1]["date"])
        history   = list(med_data["quantity_sold"].tail(30))

        results = []
        for i in range(1, horizon + 1):
            d = last_date + pd.Timedelta(days=i)
            feat = {
                "year":        d.year,
                "month":       d.month,
                "day_of_week": d.dayofweek,
                "day_of_month":d.day,
                "week_of_year":d.isocalendar()[1],
                "quarter":     (d.month - 1) // 3 + 1,
                "is_weekend":  int(d.dayofweek >= 5),
                "month_sin":   np.sin(2 * np.pi * d.month / 12),
                "month_cos":   np.cos(2 * np.pi * d.month / 12),
                "dow_sin":     np.sin(2 * np.pi * d.dayofweek / 7),
                "dow_cos":     np.cos(2 * np.pi * d.dayofweek / 7),
                "lag_7":       history[-7]  if len(history) >= 7  else np.mean(history),
                "lag_14":      history[-14] if len(history) >= 14 else np.mean(history),
                "lag_30":      history[-30] if len(history) >= 30 else np.mean(history),
                "rolling_7":   np.mean(history[-7:])  if len(history) >= 7  else np.mean(history),
                "rolling_30":  np.mean(history[-30:]) if len(history) >= 30 else np.mean(history),
            }
            pred = max(0, round(float(
                self.models[med_id].predict(
                    pd.DataFrame([feat])[self.FEATURE_COLS]
                )[0]
            )))
            results.append({"date": d.strftime("%Y-%m-%d"), "predicted_demand": pred})
            history.append(pred)
        return results

    # ── Public : recommandation de réapprovisionnement ───────
    def recommend_restock(self, sales_df, stock_df, med_id,
                          horizon=30, safety=1.2):
        preds    = self.predict_demand(sales_df, med_id, horizon)
        total    = sum(p["predicted_demand"] for p in preds)

        ms = stock_df[stock_df["medication_id"] == med_id].sort_values("period")
        current_stock = int(ms.iloc[-1]["closing_stock"]) if len(ms) > 0 else 0
        reorder_point = int(ms.iloc[-1]["reorder_point"]) if len(ms) > 0 else 20

        required  = int(total * safety)
        to_order  = max(0, required - current_stock)
        daily_avg = total / horizon if horizon > 0 else 1
        days_left = current_stock / daily_avg if daily_avg > 0 else float("inf")

        if days_left < 7:
            urgency, urgency_level = "CRITIQUE", 3
        elif days_left < 14:
            urgency, urgency_level = "URGENT", 2
        elif days_left < horizon:
            urgency, urgency_level = "MODÉRÉ", 1
        else:
            urgency, urgency_level = "OK", 0

        h1    = sum(p["predicted_demand"] for p in preds[:horizon // 2])
        h2    = sum(p["predicted_demand"] for p in preds[horizon // 2:])
        trend = "hausse" if h2 > h1 * 1.1 else ("baisse" if h2 < h1 * 0.9 else "stable")

        return {
            "medication_id":              med_id,
            "current_stock":              current_stock,
            "reorder_point":              reorder_point,
            "predicted_demand_30d":       total,
            "recommended_order_quantity": to_order,
            "required_stock":             required,
            "days_of_stock_remaining":    round(days_left, 1),
            "urgency":                    urgency,
            "urgency_level":              urgency_level,
            "demand_trend":               trend,
            "safety_factor":              safety,
            "daily_predictions":          preds[:7],
            "model_accuracy":             self.metrics.get(med_id, {}).get("accuracy", 0),
        }

    # ── Sauvegarde / Chargement ──────────────────────────────
    def save(self, path):
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        joblib.dump({"models": self.models, "metrics": self.metrics}, path)
        print(f"  ✅ Modèles sauvegardés → {path}")

    @classmethod
    def load(cls, path):
        obj  = cls()
        data = joblib.load(path)
        obj.models  = data["models"]
        obj.metrics = data["metrics"]
        return obj


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    BASE       = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR   = os.path.join(BASE, "data")
    MODELS_DIR = os.path.join(BASE, "models")
    os.makedirs(DATA_DIR,   exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    SALES_CSV  = os.path.join(DATA_DIR, "sales_history.csv")
    STOCK_CSV  = os.path.join(DATA_DIR, "stock_history.csv")
    MODEL_PKL  = os.path.join(MODELS_DIR, "demand_predictor.pkl")

    print("=" * 55)
    print("  PharmaPred — Entraînement des modèles ML")
    print("=" * 55)
    print(f"  Dossier : {BASE}")

    if not os.path.exists(SALES_CSV):
        print("\n❌ sales_history.csv introuvable.")
        print("   Lancez d'abord : python generate_data.py")
        sys.exit(1)

    sales_df = pd.read_csv(SALES_CSV)
    stock_df = pd.read_csv(STOCK_CSV)
    print(f"\n📂 Données chargées : {len(sales_df):,} enregistrements\n")

    predictor = PharmacyDemandPredictor()

    print("🤖 Entraînement en cours...")
    for med_id in sales_df["medication_id"].unique():
        df_med = sales_df[sales_df["medication_id"] == med_id].copy()
        df_med["date"]         = pd.to_datetime(df_med["date"])
        df_med                 = df_med.sort_values("date")
        df_med["year"]         = df_med["date"].dt.year
        df_med["month"]        = df_med["date"].dt.month
        df_med["day_of_week"]  = df_med["date"].dt.dayofweek
        df_med["day_of_month"] = df_med["date"].dt.day
        df_med["week_of_year"] = df_med["date"].dt.isocalendar().week.astype(int)
        df_med["quarter"]      = df_med["date"].dt.quarter
        df_med["is_weekend"]   = (df_med["day_of_week"] >= 5).astype(int)
        df_med["month_sin"]    = np.sin(2 * np.pi * df_med["month"] / 12)
        df_med["month_cos"]    = np.cos(2 * np.pi * df_med["month"] / 12)
        df_med["dow_sin"]      = np.sin(2 * np.pi * df_med["day_of_week"] / 7)
        df_med["dow_cos"]      = np.cos(2 * np.pi * df_med["day_of_week"] / 7)
        for lag in [7, 14, 30]:
            df_med[f"lag_{lag}"] = df_med["quantity_sold"].shift(lag)
        df_med["rolling_7"]  = df_med["quantity_sold"].shift(1).rolling(7).mean()
        df_med["rolling_30"] = df_med["quantity_sold"].shift(1).rolling(30).mean()
        df_med = df_med.dropna()
        if len(df_med) < 60:
            continue

        X     = df_med[PharmacyDemandPredictor.FEATURE_COLS]
        y     = df_med["quantity_sold"]
        split = int(len(df_med) * 0.8)

        model = RandomForestRegressor(
            n_estimators=100, max_depth=10,
            min_samples_split=5, random_state=42, n_jobs=-1
        )
        model.fit(X.iloc[:split], y.iloc[:split])
        y_pred = np.maximum(0, model.predict(X.iloc[split:]))
        mae    = mean_absolute_error(y.iloc[split:], y_pred)
        r2     = r2_score(y.iloc[split:], y_pred)
        acc    = round(max(0.0, 1 - mae / float(y.iloc[split:].mean())) * 100, 1)

        predictor.models[med_id]  = model
        predictor.metrics[med_id] = {
            "mae":      round(float(mae), 2),
            "rmse":     round(float(np.sqrt(mean_squared_error(y.iloc[split:], y_pred))), 2),
            "r2":       round(float(r2), 3),
            "accuracy": acc,
        }
        print(f"  {med_id}: R²={r2:.3f}  MAE={mae:.2f}  Précision={acc}%")

    avg_r2  = np.mean([m["r2"]       for m in predictor.metrics.values()])
    avg_mae = np.mean([m["mae"]      for m in predictor.metrics.values()])
    avg_acc = np.mean([m["accuracy"] for m in predictor.metrics.values()])
    print(f"\nGlobal — R²={avg_r2:.3f}  MAE={avg_mae:.2f}  Précision={avg_acc:.1f}%")

    print("\n🧪 Test de recommandation (MED001)...")
    rec = predictor.recommend_restock(sales_df, stock_df, "MED001")
    print(f"  Urgence      : {rec['urgency']}")
    print(f"  Stock actuel : {rec['current_stock']}")
    print(f"  À commander  : {rec['recommended_order_quantity']}")
    print(f"  Précision ML : {rec['model_accuracy']}%")

    print()
    predictor.save(MODEL_PKL)
    print("\n✅ Entraînement terminé — modèles prêts dans models/")
    print("   Lancez maintenant : python app.py")
    print("=" * 55)
