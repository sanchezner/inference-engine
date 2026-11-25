PY ?= .venv/bin/python
export PYTHONPATH := .
export HF_HUB_DISABLE_PROGRESS_BARS := 1
export TQDM_DISABLE := 1

.PHONY: check bringup bench-bandwidth bench-profile bench-baseline bench-paged bench-prefix bench-fusion bench-chunked

check:
	$(PY) tests/unit/paged_attn.py
	$(PY) tests/unit/reshape_cache.py
	$(PY) tests/unit/rmsnorm.py
	$(PY) tests/unit/rope.py
	$(PY) tests/unit/prefix_cache.py
	$(PY) tests/sched/sampling.py
	$(PY) tests/sched/engine_core.py
	$(PY) tests/sched/trace.py
	$(PY) checks/trace.py
	$(PY) checks/prefix.py
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

bench-profile:
	$(PY) bench/profile.py

bench-baseline:
	$(PY) bench/baseline.py

bench-paged:
	$(PY) bench/micro/paged_decode.py

bench-prefix:
	$(PY) bench/e2e/prefix.py

bench-fusion:
	$(PY) bench/fusion.py

bench-chunked:
	$(PY) bench/e2e/chunked.py
