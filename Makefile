IMAGE := vcu-surrogate:0.1.0
TEST_IMAGE := vcu-pipeline-test:0.1.0
ROOT := $(CURDIR)
GPU_RUN := docker run --rm --gpus all --ipc=host --user $$(id -u):$$(id -g) \
	-e HOME=/tmp -v "$(ROOT):/workspace" -w /workspace $(IMAGE)
RUN := docker run --rm --ipc=host --user $$(id -u):$$(id -g) \
	-e HOME=/tmp -v "$(ROOT):/workspace" -w /workspace $(IMAGE)

.PHONY: build build-test preflight download normalize infer encode metrics traces transitions validate train evaluate export-policy render-policy benchmark all test test-native

build:
	docker build -f docker/Dockerfile -t $(IMAGE) .

build-test:
	docker build --target pipeline-test -f docker/Dockerfile -t $(TEST_IMAGE) .

preflight:
	$(GPU_RUN) preflight

download:
	$(RUN) download

normalize:
	$(RUN) normalize

infer:
	$(GPU_RUN) infer

encode:
	$(RUN) encode-grid

metrics:
	$(RUN) metrics

traces:
	$(RUN) traces

transitions:
	$(RUN) transitions

validate:
	$(GPU_RUN) validate

train:
	$(GPU_RUN) train

evaluate:
	$(GPU_RUN) evaluate

export-policy:
	$(RUN) export-policy

render-policy:
	$(RUN) render-policy

# All GOP/action pairs, deterministic extra frame sampling; use BENCHMARK_FRAMES=30 for exhaustive frames.
BENCHMARK_FRAMES ?= 3
benchmark:
	$(GPU_RUN) benchmark --frames-per-gop $(BENCHMARK_FRAMES)

test:
	$(RUN) test

test-native:
	docker run --rm -e VCU_PROJECT_ROOT=/workspace \
	  -v "$(ROOT):/workspace" -w /workspace $(TEST_IMAGE) \
	  sh -lc 'PYTHONPATH=/workspace/src python3 -m compileall -q src tests && PYTHONPATH=/workspace/src python3 -m pytest -q tests'

all: preflight download normalize infer encode metrics traces transitions validate train evaluate render-policy
