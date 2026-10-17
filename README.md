# TP MLOps — entraînement reproductible

Ce projet entraîne un classificateur de régression logistique sur le jeu
Breast Cancer Wisconsin intégré à scikit-learn. Le prétraitement (imputation
et standardisation) est inclus dans le pipeline afin qu'il soit ajusté
uniquement sur les données d'entraînement.

## Structure

- `config/config.yaml` : graine aléatoire, paramètres du modèle et configuration MLflow.
- `src/train.py` : validation des données, séparation, entraînement, évaluation et suivi.
- `tests/test_pipeline.py` : tests de validation, de prédiction et de suivi MLflow.
- `artifacts/` : répertoire réservé aux artefacts générés.
- `mlruns/` : expériences MLflow locales, ignorées par Git.

Le jeu de données est fourni par scikit-learn ; aucun téléchargement de données
supplémentaire n'est nécessaire.

## Recréer l'expérience en local

Depuis la racine du dépôt, avec Python 3.10 ou ultérieur :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Les versions récentes de MLflow désactivent par défaut le backend local en
fichiers utilisé dans ce TP. Le script d'entraînement l'autorise pour conserver
la configuration `file:./mlruns`.

Lancer les tests puis l'entraînement :

```powershell
python -m pytest -q
python src/train.py
```

Afficher les expériences dans MLflow. Sous Windows, `--workers 1` évite les
erreurs de socket Uvicorn avec plusieurs processus :

```powershell
python -m mlflow ui --backend-store-uri .\mlruns --host 127.0.0.1 --port 5000 --workers 1
```

Ouvrez ensuite `http://127.0.0.1:5000`.

## Exécution avec Docker

Construire et lancer depuis la racine du dépôt. Le volume conserve les
expériences après la suppression du conteneur :

```powershell
docker build -t tp-mlops:latest .
New-Item -ItemType Directory -Force mlruns | Out-Null
$projectPath = (Get-Location).Path.Replace('\', '/')
docker run --rm -v "${projectPath}/mlruns:/app/mlruns" tp-mlops:latest
```

## Reproductibilité

La configuration consigne la graine, la taille du jeu de test et les
hyperparamètres du modèle. La séparation est stratifiée ; les métriques
accuracy, F1 et ROC AUC, les paramètres, les versions de Python et des
bibliothèques principales ainsi que le pipeline entraîné sont enregistrés
dans MLflow. Le commit Git est ajouté lorsqu'il est disponible.

Pour noter l'environnement utilisé :

```powershell
python --version
python -m pip freeze
git rev-parse HEAD
```

Les dépendances de `requirements.txt` ne sont pas épinglées. Après validation
de l'environnement, `python -m pip freeze` permet d'en conserver les versions
exactes pour reproduire cette exécution.
