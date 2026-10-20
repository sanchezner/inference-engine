PY ?= .venv/bin/python
export PYTHONPATH := .
export HF_HUB_DISABLE_PROGRESS_BARS := 1
export TQDM_DISABLE := 1

.PHONY: check bringup

check:
	$(PY) checks/forward.py
	$(PY) checks/cache.py
	$(PY) checks/batch.py
	$(PY) checks/paged.py

bringup:
	$(PY) bench/bringup.py
