# Contributing

1. Fork the repository and create a branch: `git switch -c feature/short-name`.
2. Install the tools: `pip install -r requirements-dev.txt` (CPU PyTorch is enough: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` first).
3. Make your change and add or update tests.
4. Run the same checks as CI:
   ```bash
   ruff check api tests/test_api.py && ruff format --check api tests/test_api.py
   mypy
   pytest -q
   ```
5. Open a pull request that says what changed and why.

## Ground rules

- **Evaluation honesty comes first.** Splits are by patient, the threshold is chosen on validation data only, and the external set is never used for tuning. A change that weakens any of these needs a clear reason in the pull request.
- The API tests use a randomly initialised model: they check plumbing, not accuracy. Never commit trained weights or patient data.
- Keep the README results table in step with the numbers that the code actually produces.
