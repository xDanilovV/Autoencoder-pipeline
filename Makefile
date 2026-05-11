.PHONY: install install-dev train clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

train:
	$(PYTHON) main.py

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir() and '.venv' not in p.parts and '.git' not in p.parts]"
