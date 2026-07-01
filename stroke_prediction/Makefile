.PHONY: run install docker-build docker-run test lint

run:
	streamlit run app.py

install:
	pip install -r requirements.txt

docker-build:
	docker build -t stroke-prediction-app .

docker-run:
	docker run -p 8501:8501 stroke-prediction-app

test:
	pytest tests/

lint:
	flake8 app.py src/
