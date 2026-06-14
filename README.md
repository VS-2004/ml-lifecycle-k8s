# ML Model Lifecycle Management on Kubernetes

A production-grade ML deployment pipeline built with:
- **scikit-learn** — model training
- **MLflow** — experiment tracking & model registry
- **FastAPI** — REST API serving
- **Docker** — containerization
- **Kubernetes (Minikube)** — orchestration & auto-scaling

## Phases
- Phase 1: Train model + track with MLflow
- Phase 2: Serve via FastAPI `/predict` endpoint
- Phase 3: Containerize with Docker
- Phase 4: Orchestrate on Kubernetes with HPA
