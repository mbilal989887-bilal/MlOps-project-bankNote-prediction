# Bank Note Authentication – End-to-End MLOps Project

Predicts if a bank note is genuine or fake from 4 image features
(variance, skewness, curtosis, entropy).

Tools: DVC (pipeline) · MLflow (tracking + model registry) · Flask (API + web form) · Docker · GitHub Actions

## Project structure

```
├─ data/source/      original dataset (kept in Git)
├─ src/              pipeline code (ingestion, preprocessing, training, evaluation)
├─ tests/            pipeline tests + API tests
├─ templates/ static/  web form
├─ app.py            Flask app (/, /predict, /api/predict, /health)
├─ params.yaml       ALL settings (split, model, thresholds, labels)
├─ dvc.yaml          pipeline: ingestion → preprocessing → training → evaluation
├─ Dockerfile        builds the API image
└─ .github/workflows/ci.yaml   test → docker build → smoke test → push
```

## Run on your Mac

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

git init && git add . && git commit -m "first commit"   # DVC needs a git repo
dvc repro            # runs the whole pipeline
pytest               # runs all tests
mlflow ui --backend-store-uri sqlite:///mlflow.db   # open http://127.0.0.1:5000
python app.py        # open http://127.0.0.1:8000
```

After the first `dvc repro`, commit the results:

```bash
git add dvc.lock reports models/.gitignore data/*/.gitignore
git commit -m "first pipeline run"
```

## Useful DVC commands

```bash
dvc dag                 # show the pipeline
dvc metrics show        # show metrics
dvc metrics diff        # compare with last commit
dvc params diff         # which params changed
dvc plots show          # confusion matrix plot (opens an HTML file)
```

Try it: change `training.decision_tree.max_depth` in `params.yaml`, run `dvc repro`.
Only training and evaluation run again. Set `training.model_type: random_forest` to compare models in MLflow.

## API

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"variance": 3.6, "skewness": 8.6, "curtosis": -2.8, "entropy": -0.4}'
# {"confidence":1.0,"label":"Fake","prediction":0}
```

## Docker (local)

```bash
dvc repro   # the model file must exist first
docker build -t banknote-mlops .
docker run -p 8000:8000 banknote-mlops
```

## CI/CD (GitHub Actions)

On every push to `main`: install → `dvc repro` → `pytest` → build Docker image →
smoke test the container → push to Docker Hub (tags: `latest` and the commit SHA).

Add these in GitHub → Settings → Secrets and variables → Actions:

```
DOCKERHUB_USERNAME   your Docker Hub username
DOCKERHUB_TOKEN      a Docker Hub access token (not your password)
```

## DVC remote (optional)

The dataset is small and lives in Git, so CI does not need a remote. To store
data/models in the cloud:

```bash
pip install "dvc[s3]"      # or dvc-gdrive
dvc remote add -d storage s3://your-bucket/path
dvc push
```

## Important notes

- `serving.class_labels` in `params.yaml` says 0 = Fake and 1 = Genuine
  (same as your old app). Check this against your dataset description and fix it if needed.
- The pipeline fails if test accuracy is below `evaluation.min_accuracy`.
- `mlruns/`, `mlflow.db` and `logs/` are local files and are not committed.
