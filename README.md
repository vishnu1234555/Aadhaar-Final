# Aadhaar 2 Final 🚀

A production-ready, containerized Generative AI application. This repository houses a split-stack architecture featuring a high-performance React/Vite frontend and a robust Python backend powered by PyTorch and GLiNER.

## 🏗 Architecture

* **Frontend:** Built with Vite & React, served via an optimized Nginx Alpine container.
* **Backend:** Python 3.10 slim environment running FastAPI, utilizing `torch==2.5.1+cpu` for lightweight, GPU-independent inference.
* **Deployment:** Fully containerized via Docker Compose with strict build contexts and cache-optimized layer management.

## ⚙️ Quick Start (Docker)

Ensure Docker Desktop is running, then execute the following command at the root of the repository to build and spin up the containers:

```bash
docker compose up --build
