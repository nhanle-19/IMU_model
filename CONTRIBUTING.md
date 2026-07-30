# Contributing to TartanIMU

Thank you for helping improve TartanIMU. This guide defines the development,
testing, and compatibility standards for contributions.

## Development Setup

TartanIMU requires Python 3.10 or newer.

```bash
git clone https://github.com/superxslam/TartanIMU.git
cd TartanIMU
pip install -e ".[logging,dev]"
```

Before submitting a change, run:

```bash
ruff check tartan_imu/
pytest tests/unit -q
```

## Code Style

- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
- Keep code `ruff`-clean and within the configured 88-character line length.
- Add Google-style docstrings to public modules, classes, and functions.
- Type-annotate public function and method signatures.
- Group imports into standard library, third-party, and first-party sections.
- Use the project logger instead of debug `print` statements.

## Naming

| Kind | Convention | Example |
| --- | --- | --- |
| Modules and packages | `lower_snake_case` | `model_factory.py` |
| Classes | `PascalCase` | `FoundationModel` |
| Functions and variables | `lower_snake_case` | `build_backbone` |
| Constants | `UPPER_SNAKE_CASE` | `GRAVITY` |
| Internal helpers | Leading underscore | `_create_model` |
| Registry keys | Match the configuration value | `Foundation_Model` |

Names should describe intent. Domain-standard abbreviations such as IMU, cov,
and seq are acceptable; avoid ambiguous abbreviations.

## Design Guidelines

- Keep modules and functions focused on one responsibility.
- Remove dead code and commented-out implementation blocks.
- Put shared model logic in `tartan_imu/model/common/` and general helpers in
  `tartan_imu/utils/`.
- Read tunable values from configuration instead of introducing magic numbers.
- Add model backbones through a registered builder in
  `tartan_imu/model/backbones/`, not another dispatch chain.
- Add dataset readers under `tartan_imu/dataloader/` and register them in
  `tartan_imu/utils/registry.py`.

## Checkpoint Compatibility

PyTorch module attribute names map directly to checkpoint `state_dict` keys.
Renaming attributes such as `self.trunk` or `self.head` can make released
checkpoints incompatible.

When changing model structure:

1. Preserve existing attribute names whenever possible.
2. Add an explicit migration path when a rename is unavoidable.
3. Add a characterization test that pins affected parameter names and shapes.
4. Verify loading with a representative released checkpoint.

## Tests

- Every bug fix and feature should include focused test coverage.
- Use characterization tests for refactors that must preserve numerical
  behavior or parameter names.
- Keep tests deterministic and independent of unavailable private datasets.
- Run the complete unit suite before opening a pull request.
- Report any skipped or unavailable checks in the pull request description.

## Commits and Pull Requests

- Keep commits small and focused.
- Use an imperative subject with an optional scope, for example
  `fix(data): reject empty trajectories`.
- Do not include AI-authorship trailers.
- Do not force-push shared branches or overwrite another contributor's work.
- Explain user-visible behavior, compatibility impact, and verification in the
  pull request description.

## Licensing

By contributing, you agree that your contribution will be licensed under the
repository's [Apache License 2.0](LICENSE).
