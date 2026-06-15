# ML Model Lifecycle Management on Kubernetes

A production-grade ML deployment pipeline built from scratch.

## Tech Stack
- **scikit-learn** — RandomForest classification model
- **MLflow** — experiment tracking & model registry
- **FastAPI** — REST API with `/predict` and `/health` endpoints
- **Docker** — self-contained containerization
- **Kubernetes (Minikube)** — orchestration with auto-scaling

## Architecture
Train (scikit-learn) → Track (MLflow) → Serve (FastAPI) → Containerize (Docker) → Orchestrate (Kubernetes)

## Project Structure
ml-lifecycle-k8s/

├── data/              # dataset generation

├── train/             # model training + MLflow logging

├── app/               # FastAPI application

│   ├── main.py

│   ├── model_loader.py

│   └── schemas.py

├── k8s/               # Kubernetes manifests

│   ├── deployment.yaml

│   ├── service.yaml

│   └── hpa.yaml

├── Dockerfile

└── requirements.txt

## Quick Start

### 1. Train the model
```bash
python data/generate_data.py
python train/train.py
```

### 2. Run locally
```bash
python -m uvicorn app.main:app --reload --port 8000
```

### 3. Run with Docker
```bash
docker build -t ml-api:v1 .
docker run -p 8000:8000 ml-api:v1
```

### 4. Deploy on Kubernetes
```bash
minikube start
eval $(minikube docker-env)
docker build -t ml-api:v1 .
kubectl apply -f k8s/
minikube service ml-api-service --url
```

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Get prediction |
| GET | `/docs` | Swagger UI |

## Results
- Model accuracy: ~91% on test set
- API throughput: ~200 req/s on single pod
- Auto-scales to 3 replicas under load via HPA
