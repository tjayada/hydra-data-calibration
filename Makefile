DATASET ?= 24_12_19_hydra_icp_eval
OUT     ?= hydra_eval

.PHONY: download calibrate dataset lint

download:
	python download_data.py

calibrate:
	python calibrate.py --dataset $(DATASET)

dataset:
	python build_dataset.py --dataset $(DATASET) --out $(OUT)

lint:
	python -m ruff check .
