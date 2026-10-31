PY ?= .venv/bin/python
export PYTHONPATH := .
export HF_HUB_DISABLE_PROGRESS_BARS := 1
export TQDM_DISABLE := 1

.PHONY: check bringup bench-bandwidth

check:
	$(PY) checks/config.py
	$(PY) checks/roofline.py
	$(PY) checks/weights.py
	$(PY) checks/forward.py
	$(PY) checks/cache.py
	$(PY) checks/batch.py
	$(PY) checks/paged.py
	$(PY) checks/scale.py

bringup:
	$(PY) bench/bringup.py

bench-bandwidth:
	$(PY) bench/bandwidth.py
