"""
PharmaPred — Génération des données historiques
30 médicaments · 5 ans (2020-2025) · Saisonnalité réaliste
Lancez : python generate_data.py
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

# ── Catalogue des 30 médicaments ──────────────────────────────
MEDICATIONS = [
    {"id":"MED001","name":"Paracétamol 500mg",    "category":"Antalgique",         "unit_cost":2.5,  "reorder_point":50},
    {"id":"MED002","name":"Amoxicilline 500mg",   "category":"Antibiotique",       "unit_cost":8.0,  "reorder_point":30},
    {"id":"MED003","name":"Ibuprofène 400mg",     "category":"Anti-inflammatoire", "unit_cost":3.5,  "reorder_point":40},
    {"id":"MED004","name":"Oméprazole 20mg",      "category":"Gastrique",          "unit_cost":5.0,  "reorder_point":25},
    {"id":"MED005","name":"Metformine 850mg",     "category":"Diabète",            "unit_cost":4.0,  "reorder_point":35},
    {"id":"MED006","name":"Amlodipine 5mg",       "category":"Cardiovasculaire",   "unit_cost":6.5,  "reorder_point":20},
    {"id":"MED007","name":"Salbutamol Spray",     "category":"Respiratoire",       "unit_cost":12.0, "reorder_point":15},
    {"id":"MED008","name":"Loratadine 10mg",      "category":"Antihistaminique",   "unit_cost":4.5,  "reorder_point":30},
    {"id":"MED009","name":"Vitamine C 1000mg",    "category":"Supplément",         "unit_cost":3.0,  "reorder_point":45},
    {"id":"MED010","name":"Doliprane 1000mg",     "category":"Antalgique",         "unit_cost":3.0,  "reorder_point":60},
    {"id":"MED011","name":"Augmentin 875mg",      "category":"Antibiotique",       "unit_cost":15.0, "reorder_point":20},
    {"id":"MED012","name":"Ventoline 100mcg",     "category":"Respiratoire",       "unit_cost":14.0, "reorder_point":15},
    {"id":"MED013","name":"Atorvastatine 20mg",   "category":"Cardiovasculaire",   "unit_cost":7.0,  "reorder_point":25},
    {"id":"MED014","name":"Lévothyroxine 50mcg",  "category":"Thyroïde",           "unit_cost":5.5,  "reorder_point":20},
    {"id":"MED015","name":"Pantoprazole 40mg",    "category":"Gastrique",          "unit_cost":6.0,  "reorder_point":30},
    {"id":"MED016","name":"Sertraline 50mg",      "category":"Psychiatrique",      "unit_cost":9.0,  "reorder_point":15},
    {"id":"MED017","name":"Lansoprazole 30mg",    "category":"Gastrique",          "unit_cost":5.5,  "reorder_point":20},
    {"id":"MED018","name":"Cétirizine 10mg",      "category":"Antihistaminique",   "unit_cost":3.5,  "reorder_point":35},
    {"id":"MED019","name":"Diclofénac 50mg",      "category":"Anti-inflammatoire", "unit_cost":4.0,  "reorder_point":30},
    {"id":"MED020","name":"Tramadol 50mg",        "category":"Antalgique",         "unit_cost":6.0,  "reorder_point":20},
    {"id":"MED021","name":"Fluconazole 150mg",    "category":"Antifongique",       "unit_cost":10.0, "reorder_point":15},
    {"id":"MED022","name":"Ciprofloxacine 500mg", "category":"Antibiotique",       "unit_cost":12.0, "reorder_point":20},
    {"id":"MED023","name":"Doxycycline 100mg",    "category":"Antibiotique",       "unit_cost":8.5,  "reorder_point":20},
    {"id":"MED024","name":"Prednisolone 5mg",     "category":"Corticoïde",         "unit_cost":4.5,  "reorder_point":25},
    {"id":"MED025","name":"Vitamine D3 1000UI",   "category":"Supplément",         "unit_cost":3.5,  "reorder_point":40},
    {"id":"MED026","name":"Magnésium 300mg",      "category":"Supplément",         "unit_cost":4.0,  "reorder_point":35},
    {"id":"MED027","name":"Fer 80mg",             "category":"Supplément",         "unit_cost":3.0,  "reorder_point":30},
    {"id":"MED028","name":"Acide Folique 5mg",    "category":"Supplément",         "unit_cost":2.5,  "reorder_point":25},
    {"id":"MED029","name":"Bisoprolol 5mg",       "category":"Cardiovasculaire",   "unit_cost":5.0,  "reorder_point":20},
    {"id":"MED030","name":"Ramipril 5mg",         "category":"Cardiovasculaire",   "unit_cost":5.5,  "reorder_point":20},
]

# Facteurs saisonniers par catégorie (12 mois)
SEASONAL = {
    "Antalgique":         [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.1,1.2,1.3,1.2],
    "Antibiotique":       [1.5,1.3,1.1,0.9,0.8,0.7,0.7,0.8,1.0,1.2,1.4,1.6],
    "Anti-inflammatoire": [1.2,1.1,1.0,1.0,0.9,0.9,0.9,1.0,1.1,1.2,1.2,1.3],
    "Gastrique":          [1.0,1.0,1.0,1.1,1.2,1.1,1.0,1.0,1.0,1.0,1.0,1.0],
    "Diabète":            [1.0]*12,
    "Cardiovasculaire":   [1.1,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.1,1.1],
    "Respiratoire":       [1.6,1.4,1.1,0.9,0.7,0.6,0.6,0.7,0.9,1.1,1.3,1.5],
    "Antihistaminique":   [0.7,0.7,1.0,1.5,2.0,1.8,1.5,1.2,1.0,0.8,0.7,0.7],
    "Supplément":         [1.2,1.1,1.0,1.0,0.9,0.8,0.8,0.9,1.0,1.1,1.2,1.3],
    "Thyroïde":           [1.0]*12,
    "Psychiatrique":      [1.0,1.0,1.0,1.0,1.0,0.9,0.9,1.0,1.0,1.0,1.1,1.1],
    "Antifongique":       [0.9,0.9,1.0,1.1,1.2,1.3,1.3,1.2,1.1,1.0,0.9,0.9],
    "Corticoïde":         [1.3,1.2,1.0,1.0,0.9,0.8,0.8,0.9,1.0,1.1,1.2,1.3],
}

BASE_DEMANDS = {
    "MED001":25,"MED002":12,"MED003":18,"MED004":10,"MED005":8, "MED006":6,
    "MED007":5, "MED008":9, "MED009":15,"MED010":30,"MED011":7, "MED012":4,
    "MED013":7, "MED014":5, "MED015":8, "MED016":4, "MED017":6, "MED018":11,
    "MED019":9, "MED020":5, "MED021":3, "MED022":6, "MED023":5, "MED024":7,
    "MED025":12,"MED026":10,"MED027":8, "MED028":6, "MED029":5, "MED030":5,
}

def generate_sales():
    records = []
    start = datetime(2020, 1, 1)
    end   = datetime(2025, 3, 31)
    cur   = start
    while cur <= end:
        for med in MEDICATIONS:
            base    = BASE_DEMANDS[med["id"]]
            sf      = SEASONAL.get(med["category"], [1.0]*12)[cur.month - 1]
            wf      = [1.1, 1.1, 1.0, 1.0, 1.2, 1.0, 0.5][cur.weekday()]
            trend   = 1.0 + (cur - start).days / 365.0 * 0.03
            demand  = base * sf * wf * trend
            noise   = np.random.normal(0, demand * 0.15)
            qty     = max(0, int(demand + noise))
            if qty > 0:
                records.append({
                    "date":            cur.strftime("%Y-%m-%d"),
                    "medication_id":   med["id"],
                    "medication_name": med["name"],
                    "category":        med["category"],
                    "quantity_sold":   qty,
                    "unit_price":      round(med["unit_cost"] * 1.3, 2),
                    "revenue":         round(qty * med["unit_cost"] * 1.3, 2),
                })
        cur += timedelta(days=1)
    return pd.DataFrame(records)

def generate_stock(sales_df):
    records = []
    for med in MEDICATIONS:
        ms      = sales_df[sales_df["medication_id"] == med["id"]]
        monthly = ms.groupby(
            pd.to_datetime(ms["date"]).dt.to_period("M")
        )["quantity_sold"].sum()
        stock = med["reorder_point"] * 5
        for period, sold in monthly.items():
            restock = med["reorder_point"] * 6 if stock < med["reorder_point"] * 2 else 0
            stock   = max(0, stock + restock - sold)
            records.append({
                "period":             str(period),
                "medication_id":      med["id"],
                "medication_name":    med["name"],
                "category":           med["category"],
                "opening_stock":      stock + int(sold) - restock,
                "quantity_sold":      int(sold),
                "quantity_restocked": restock,
                "closing_stock":      stock,
                "reorder_point":      med["reorder_point"],
                "unit_cost":          med["unit_cost"],
            })
    return pd.DataFrame(records)

if __name__ == "__main__":
    # Utilise le dossier du script — fonctionne partout (double-clic, terminal, IDE)
    BASE       = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR   = os.path.join(BASE, "data")
    MODELS_DIR = os.path.join(BASE, "models")

    os.makedirs(DATA_DIR,   exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 52)
    print("  PharmaPred — Génération des données")
    print("=" * 52)
    print(f"  Dossier : {BASE}")

    print("\n📊 Génération des ventes journalières...")
    sales_df = generate_sales()
    sales_df.to_csv(os.path.join(DATA_DIR, "sales_history.csv"), index=False)
    print(f"  ✅ {len(sales_df):,} enregistrements | "
          f"{sales_df['date'].min()} → {sales_df['date'].max()}")

    print("\n📦 Génération des données de stock...")
    stock_df = generate_stock(sales_df)
    stock_df.to_csv(os.path.join(DATA_DIR, "stock_history.csv"), index=False)
    print(f"  ✅ {len(stock_df):,} entrées de stock mensuel")

    print("\n💊 Sauvegarde du catalogue médicaments...")
    pd.DataFrame(MEDICATIONS).to_csv(os.path.join(DATA_DIR, "medications.csv"), index=False)
    print(f"  ✅ {len(MEDICATIONS)} médicaments — 13 catégories")

    print("\n" + "=" * 52)
    print(f"  CA total       : {sales_df['revenue'].sum():>12,.0f} MAD")
    print(f"  Unités totales : {sales_df['quantity_sold'].sum():>12,}")
    print("=" * 52)
    print(f"\n✅ Données prêtes dans : {DATA_DIR}")
