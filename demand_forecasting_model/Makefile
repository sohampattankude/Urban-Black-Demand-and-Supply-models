.PHONY: help setup install download preprocess train test api clean logs

help:
	@echo "Demand Forecasting Model - Development Commands"
	@echo "==============================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup        - Create virtual environment and install dependencies"
	@echo "  make install      - Install package in development mode"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make download     - Download data from BigQuery warehouse"
	@echo "  make preprocess   - Run feature engineering on raw data"
	@echo ""
	@echo "Model Training:"
	@echo "  make train        - Train LSTM + XGBoost ensemble models"
	@echo "  make evaluate     - Generate evaluation metrics"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test         - Run unit tests with pytest"
	@echo "  make coverage     - Generate test coverage report"
	@echo "  make lint         - Run code quality checks (flake8)"
	@echo "  make format       - Format code with black"
	@echo ""
	@echo "Serving:"
	@echo "  make api          - Start REST API server (port 8000)"
	@echo "  make api-prod     - Start API in production mode"
	@echo ""
	@echo "Development:"
	@echo "  make notebook     - Start Jupyter notebook server"
	@echo "  make logs         - View recent logs"
	@echo "  make clean        - Remove cached files and artifacts"
	@echo ""

setup:
	python -m venv venv
	. venv/Scripts/activate && pip install --upgrade pip
	. venv/Scripts/activate && pip install -r requirements.txt
	mkdir -p data/{raw,processed,time_based_indices}
	mkdir -p models logs plots
	@echo "✅ Setup complete. Run 'venv\\Scripts\\activate' to activate environment."

install:
	pip install -e .

download:
	@echo "Downloading data from BigQuery..."
	python scripts/download_data.py --days 180

preprocess:
	@echo "Running feature engineering..."
	python scripts/preprocess.py

train:
	@echo "Training models (LSTM + XGBoost)..."
	python scripts/train.py

evaluate:
	@echo "Evaluating model performance..."
	python scripts/evaluate.py

test:
	@echo "Running unit tests..."
	pytest tests/ -v --tb=short

coverage:
	@echo "Generating coverage report..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report generated: htmlcov/index.html"

lint:
	@echo "Running code quality checks..."
	flake8 src/ --max-line-length=100
	pylint src/ --disable=unnecessary-comprehension

format:
	@echo "Formatting code with black..."
	black src/ tests/ scripts/

api:
	@echo "Starting API server (dev mode)..."
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

api-prod:
	@echo "Starting API server (production mode)..."
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --workers 4

notebook:
	@echo "Starting Jupyter notebook..."
	jupyter notebook notebooks/

logs:
	@echo "Recent logs:"
	@tail -50 logs/*.log

clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
	find . -type f -name '*.pyc' -delete
	find . -type f -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null
	find . -type f -name '.coverage' -delete
	rm -rf htmlcov/ dist/ build/ *.egg-info
	@echo "✅ Cleanup complete"

# Full pipeline
pipeline: download preprocess train evaluate
	@echo "✅ Full pipeline complete"

# Development workflow
dev: format lint test
	@echo "✅ Development checks complete"
