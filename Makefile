.DEFAULT_GOAL := setup

.PHONY: .uv
.uv:
	@uv -V || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: setup
setup: .uv
	uv sync --frozen

.PHONY: check-rocm
check-rocm:
	@uv run python -c "import torch; print(f'Version: {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}')"
