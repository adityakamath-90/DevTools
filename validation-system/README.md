# Validation System

This directory contains scripts and configuration for validating generated Kotlin tests and runtime behavior.

## Structure
- `config/validation.conf` — Validation configuration
- `docs/RUNTIME_VALIDATION_GUIDE.md` — Guide to runtime validation
- `gradle-project/` — Gradle project for compiling and running tests
- `reports/validation_results.json` — Validation results
- `scripts/` — Validation scripts

## Usage

### 1. Run Validation
```sh
bash scripts/validate_runtime.sh
```

### 2. Demo Validation
```sh
bash scripts/demo_validation.sh
```

### 3. Python Validation
```sh
python scripts/validate_tests.py
```

## Notes
- Ensure all dependencies are installed and the Gradle project is set up before running validation.
- See the guide in `docs/RUNTIME_VALIDATION_GUIDE.md` for more details.
