.PHONY: tests

bash:
	scripts/collect.sh
	scripts/preprocessed.sh
	scripts/train.sh



tests:
	pytest tests/test_collect.py && \
	pytest tests/test_preprocessed.py && \
	pytest tests/test_model.py

all: 