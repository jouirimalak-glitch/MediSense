# MediSense

**Prédiction de la demande de médicaments et recommandation de réapprovisionnement par Machine Learning.**

Projet académique — Master Informatique et Télécommunications, Université Mohammed V de Rabat, Faculté des Sciences de Rabat (2025–2026).

Réalisé par **Jouiri Malak** et **Icharli Rania**, encadré par **Abdelhak Mahmoudi** (encadrant) et **Saad Frihi** (co-encadrant).

---

## Sommaire

- [Contexte et problématique](#contexte-et-problématique)
- [Objectifs du projet](#objectifs-du-projet)
- [Présentation de la solution](#présentation-de-la-solution)
- [Les données](#les-données)
- [Le modèle Machine Learning](#le-modèle-machine-learning)
- [Logique de recommandation](#logique-de-recommandation)
- [Installation et lancement](#installation-et-lancement)
- [L'API REST](#lapi-rest)
- [L'application web](#lapplication-web)
- [Résultats obtenus](#résultats-obtenus)
- [Perspectives d'évolution](#perspectives-dévolution)
- [Licence](#licence)

---

## Contexte et problématique

La gestion des stocks dans le secteur pharmaceutique constitue un défi opérationnel quotidien. Une pharmacie doit constamment équilibrer deux risques contradictoires :

- **La rupture de stock** : le médicament n'est plus disponible quand le patient en a besoin.
- **Le surstock** : des quantités excessives commandées par précaution immobilisent du capital et exposent les médicaments à la péremption.

Dans la majorité des officines, les décisions de réapprovisionnement restent fondées sur l'expérience du pharmacien plutôt que sur une analyse systématique de l'historique de ventes. Pourtant, ces établissements accumulent chaque jour des données précieuses qui restent largement inexploitées faute d'outil adapté.

MediSense part de ce constat : il est possible de transformer cet historique en prévisions fiables et en recommandations concrètes, à condition de disposer d'un modèle de prédiction adapté et d'une interface accessible aux utilisateurs non techniques.

---

## Objectifs du projet

Le projet poursuit cinq objectifs principaux :

1. **Analyser** le contexte métier et les données liées à la prédiction de la demande pharmaceutique.
2. **Préparer et structurer** des données historiques de stock et de ventes représentatives d'une officine réelle.
3. **Développer** des modèles de prévision de la demande et de recommandation de réapprovisionnement.
4. **Implémenter** un service API exposant ces fonctionnalités pour la prédiction et l'aide à la décision.
5. **Évaluer expérimentalement** l'approche proposée à l'aide de métriques reconnues (R², MAE, précision).

Un sixième objectif, ajouté en valeur supplémentaire, a été la création d'une interface web complète (tableau de bord, recommandations interactives, assistant conversationnel) afin de rendre le système utilisable sans connaissance technique préalable.

---

## Présentation de la solution

MediSense est composé de quatre couches indépendantes :

| Couche | Rôle |
|---|---|
| **Données** | Génère ou charge l'historique de ventes, de stock et le catalogue de médicaments. |
| **Apprentissage** | Entraîne un modèle Random Forest indépendant pour chacun des 30 médicaments. |
| **Service** | Expose les prédictions et recommandations via une API REST Flask. |
| **Présentation** | Tableau de bord web interactif consommant l'API, avec assistant conversationnel intégré. |

Le système couvre l'ensemble du cycle, de la donnée brute jusqu'à la décision opérationnelle, sans nécessiter d'infrastructure complexe.

---

## Les données

### Vue d'ensemble

- 57 504 enregistrements de ventes journalières
- 30 médicaments suivis
- 13 catégories thérapeutiques
- Période couverte : janvier 2020 à mars 2025 (5 ans)
- 553 313 unités vendues au total
- 3 649 803 MAD de chiffre d'affaires cumulé

### Origine des données

En l'absence d'accès à des données réelles de pharmacie pour des raisons de confidentialité, un jeu de données synthétique mais réaliste a été généré. La génération applique, pour chaque médicament et chaque jour, la formule suivante :

```
demande_du_jour = demande_de_base
                   x facteur_saisonnier[categorie, mois]
                   x facteur_weekend[jour_semaine]
                   x (1 + annees_ecoulees x 0.03)
                   + bruit_gaussien(ecart-type = 15% de la demande)
```

Cette approche garantit que chaque médicament suit un profil de consommation cohérent avec sa catégorie thérapeutique, tout en introduisant une variabilité aléatoire comparable à celle observée en conditions réelles.

### Saisonnalité modélisée

- Antibiotiques : pic hivernal entre décembre et février (jusqu'à +60 %)
- Antihistaminiques : pic printanier entre avril et juin (jusqu'à +100 %)
- Voies respiratoires : hausse hivernale d'environ 60 %
- Toutes catégories : réduction d'environ 50 % des ventes le week-end
- Tendance générale : croissance moyenne de 3 % par an sur l'ensemble du catalogue

### Catalogue des médicaments

Le catalogue comprend 30 médicaments répartis sur 13 catégories, choisies pour représenter la diversité d'une officine généraliste : Antalgique, Antibiotique, Anti-inflammatoire, Gastrique, Diabète, Cardiovasculaire, Respiratoire, Antihistaminique, Supplément, Thyroïde, Psychiatrique, Antifongique et Corticoïde.

### Fichiers de données

| Fichier | Volume | Colonnes principales |
|---|---|---|
| sales_history.csv | 57 504 lignes | date, medication_id, medication_name, category, quantity_sold, unit_price, revenue |
| stock_history.csv | 1 890 lignes | period, medication_id, opening_stock, quantity_sold, quantity_restocked, closing_stock, reorder_point |
| medications.csv | 30 lignes | id, name, category, unit_cost, reorder_point |

---

## Le modèle Machine Learning

### Choix de l'algorithme

L'algorithme retenu est le Random Forest Regressor, implémenté via la bibliothèque scikit-learn. Ce choix s'appuie sur plusieurs considérations pratiques :

- Absence de besoin de normalisation des variables d'entrée, contrairement à des modèles comme les réseaux de neurones.
- Robustesse au surapprentissage grâce à l'agrégation de multiples arbres de décision entraînés sur des sous-échantillons.
- Capacité à gérer des variables mixtes (temporelles, cycliques, retardées) sans pré-traitement complexe.
- Possibilité d'analyser l'importance relative de chaque variable explicative.

Un modèle indépendant est entraîné pour chacun des 30 médicaments, plutôt qu'un unique modèle global, afin de capturer les particularités de consommation propres à chaque produit.

### Hyperparamètres

| Paramètre | Valeur |
|---|---|
| n_estimators | 100 arbres |
| max_depth | 10 |
| min_samples_split | 5 |
| Répartition train / test | 80 % / 20 % (split chronologique) |

### Variables explicatives (features)

Douze variables sont calculées à partir de la date et de l'historique de ventes, regroupées en quatre familles :

| Famille | Variables |
|---|---|
| Temporelles | year, month, day_of_week, day_of_month, week_of_year, quarter, is_weekend |
| Cycliques | month_sin, month_cos, dow_sin, dow_cos (encodage trigonométrique évitant les discontinuités de calendrier) |
| Mémoire (lags) | lag_7, lag_14, lag_30 — valeurs de demande observées 7, 14 et 30 jours auparavant |
| Tendance (rolling) | rolling_7, rolling_30 — moyennes mobiles sur 7 et 30 jours |

---

## Logique de recommandation

À partir des prédictions de demande, le système calcule automatiquement une recommandation de commande :

```
Stock_requis = Demande_prevue_horizon x Facteur_securite (1.2 par defaut)
Quantite_a_commander = MAX( 0 ; Stock_requis - Stock_actuel )
```

Le niveau d'urgence est déterminé selon le nombre de jours de couverture restants :

| Niveau | Condition | Action attendue |
|---|---|---|
| CRITIQUE | Moins de 7 jours de stock restant | Commande immédiate |
| URGENT | Entre 7 et 14 jours | Commande prioritaire |
| MODÉRÉ | Entre 14 jours et l'horizon choisi | Commande à planifier |
| SUFFISANT | Stock couvrant l'horizon complet | Aucune action requise |

Le facteur de sécurité par défaut (x1,2) a été calibré empiriquement pour absorber la variabilité résiduelle du modèle sans provoquer de surstock systématique. Ce paramètre reste ajustable par l'utilisateur final, aussi bien via l'interface web que via l'API.

---

## Installation et lancement

### Prérequis

- Python 3.9 ou supérieur
- pip

### Étapes

```bash
git clone <url-du-depot>
cd medisense

pip install -r requirements.txt

python train_model.py

python app.py
# disponible sur http://localhost:5000
```

Ouvrir ensuite pharmacy_dashboard.html dans un navigateur pour accéder au tableau de bord.

### Connexion au tableau de bord

| Profil | Identifiant | Mot de passe |
|---|---|---|
| Administrateur | admin | pharma2026 |
| Pharmacien | pharmacien | pharma123 |
| Gestionnaire de stock | gestionnaire | stock456 |

---

## L'API REST

Le service est exposé via une API REST développée avec le framework Flask (Python). Toutes les réponses sont retournées au format JSON, avec gestion du CORS afin de permettre une consommation directe depuis l'interface web.

### Liste des points d'accès

| Méthode | Endpoint | Description |
|---|---|---|
| GET | / | Documentation générale de l'API |
| GET | /api/medications | Liste des 30 médicaments avec stock courant |
| GET | /api/medications/:id | Détail et historique mensuel d'un médicament |
| GET | /api/predict/:id?horizon=30 | Prédictions journalières sur un horizon (1 à 90 jours) |
| GET | /api/recommend/:id | Recommandation de réapprovisionnement pour un produit |
| GET | /api/recommend/all | Toutes les recommandations, triées par urgence |
| GET | /api/analytics/sales | Statistiques de ventes mensuelles et annuelles |
| GET | /api/analytics/top-movers | Classement des médicaments par volume |
| GET | /api/model/metrics | Métriques R², MAE et précision des 30 modèles |
| POST | /api/predict/custom | Prédiction avec horizon et facteur de sécurité ajustables |

### Exemple d'appel

Requête pour obtenir la recommandation de réapprovisionnement du Doliprane 1000 mg :

```
GET /api/recommend/MED010
```

Réponse :

```json
{
  "medication_id": "MED010",
  "current_stock": 0,
  "predicted_demand_30d": 976,
  "recommended_order_quantity": 1171,
  "urgency": "CRITIQUE",
  "estimated_cost": 3513.0
}
```

---

## L'application web

Au-delà du service API, une interface web autonome (pharmacy_dashboard.html) a été développée afin de rendre le système directement utilisable par un pharmacien sans connaissance technique.

### Fonctionnalités principales

- Tableau de bord : indicateurs en temps réel, graphique interactif (courbe, barres, aire), stock modifiable directement dans le tableau.
- Recommandations : 30 fiches classées par urgence, avec recalcul instantané de la quantité à commander et du coût lorsque le stock est ajusté.
- Prédictions : choix du médicament, de l'horizon et du facteur de sécurité, avec simulation à la volée.
- Analytique : historique sur cinq ans, classement des médicaments par volume et par chiffre d'affaires.
- Connexion sécurisée : trois profils utilisateurs distincts (administrateur, pharmacien, gestionnaire de stock).
- SenseBot : assistant conversationnel répondant en français à des questions en langage naturel sur le stock, les prévisions et les coûts (par exemple : "stock doliprane", "coût total commande", "médicaments critiques").

### Choix technologiques du front-end

L'interface est construite en HTML5, CSS3 et JavaScript natif, sans framework lourd, afin de rester un fichier unique facilement distribuable. La bibliothèque Chart.js assure le rendu des graphiques interactifs.

---

## Résultats obtenus

### Performances globales

| Métrique | Valeur |
|---|---|
| Modèles entraînés | 30 |
| R² moyen | 0,680 |
| MAE moyen | 1,42 unité |
| Précision moyenne | 85,6 % |

### Résultats détaillés (échantillon)

| Médicament | Catégorie | R² | MAE | Précision |
|---|---|---|---|---|
| Loratadine 10mg | Antihistaminique | 0,856 | 1,53 | 86,1 % |
| Salbutamol Spray | Respiratoire | 0,825 | 0,79 | 85,3 % |
| Cétirizine 10mg | Antihistaminique | 0,829 | 1,91 | 85,7 % |
| Paracétamol 500mg | Antalgique | 0,680 | 3,68 | 87,5 % |
| Doliprane 1000mg | Antalgique | 0,659 | 4,67 | 86,8 % |
| Sertraline 50mg | Psychiatrique | 0,553 | 0,64 | 83,9 % |
| Fluconazole 150mg | Antifongique | 0,617 | 0,56 | 81,9 % |

### Analyse

Les catégories présentant la saisonnalité la plus marquée (antihistaminiques, voies respiratoires) obtiennent les meilleurs scores R², ce qui confirme que le modèle capte efficacement les cycles prévisibles. À l'inverse, les produits à demande plus erratique (psychiatrique, antifongique) affichent un R² plus faible, bien que leur MAE reste basse en valeur absolue du fait de volumes de vente plus réduits.

Sur le jeu de données étudié, l'application des recommandations générées représente un montant total de réapprovisionnement estimé à 53 295 MAD pour couvrir l'ensemble des 30 médicaments suivis.

---

## Perspectives d'évolution

Plusieurs pistes restent ouvertes pour un passage à un usage réel en officine :

- Données en temps réel — connecter l'application à un système de caisse réel pour une mise à jour automatique du stock.
- Comparaison d'algorithmes — évaluer XGBoost et des réseaux récurrents (LSTM) face au Random Forest actuel.
- Alertes automatiques — notifier par courriel ou SMS lorsque le niveau critique est atteint sur un médicament.
- Déploiement en production — héberger l'API et le dashboard sur un serveur accessible depuis plusieurs officines.
- Extension multi-pharmacies — adapter le modèle pour suivre simultanément plusieurs points de vente.
- Intégration aux logiciels existants — connecter MediSense aux outils de gestion déjà utilisés sur le terrain.

---

## Licence

Projet académique réalisé dans le cadre du Master Informatique et Télécommunications 2025-2026, Université Mohammed V de Rabat, Faculté des Sciences de Rabat.
