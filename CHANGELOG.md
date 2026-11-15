# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-15

### Added
- FastAPI inference service: `/predict` (probability, label, optional Grad-CAM overlay), `/model`, `/health`.
- Upload validation, optional API key, safe checkpoint loading, configurable threshold.
- Docker image (CPU PyTorch, non-root, read-only) and Compose file; weights are mounted, not baked in.
- CI: ruff, strict mypy on the API, tests, and a Docker smoke test that calls `/predict`.
- CodeQL, Dependabot and a tag-driven release workflow.

## [0.1.0] - 2026-11-13

### Added
- Patient-level split, DenseNet121 training, evaluation with a validation-chosen threshold.
- Grad-CAM, shortcut (masking) check and a Gradio demo.
