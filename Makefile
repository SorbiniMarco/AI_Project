install:
	python -m pip install --upgrade pip && pip install -r requirements.txt
	@echo "Installation complete.  You can now run the project."

lint:
	PYTHONPATH=. pylint --disable=R,C src/*.py tests/*.py
	@echo "Linting complete. No issues found"

test:
	PYTHONPATH=. python -m pytest -vv --cov=src tests/
	@echo "Testing complete. All tests passed."

build:
	python -m build
	@echo "Build complete. Check dist/ directory."

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache __pycache__
	@echo "Clean complete."