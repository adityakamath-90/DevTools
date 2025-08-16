# 🧪 Validation System

A robust validation framework for ensuring the quality and correctness of generated Kotlin tests and their runtime behavior.

## 📂 Directory Structure

```
validation-system/
├── config/                    # Configuration files
│   └── validation.conf       # Main validation configuration
├── docs/                     # Documentation
│   └── RUNTIME_VALIDATION_GUIDE.md  # Detailed validation guide
├── gradle-project/           # Gradle project for test execution
│   ├── src/
│   └── build.gradle.kts
├── reports/                  # Generated reports
│   └── validation_results.json
└── scripts/                  # Validation scripts
    ├── validate_runtime.sh   # Main validation script
    ├── demo_validation.sh    # Demo validation
    └── validate_tests.py     # Python validation utilities
```

## 🚀 Quick Start

### Prerequisites

- Java 11 or higher
- Gradle 7.0+
- Python 3.9+
- Dependencies from `requirements.txt`

### Installation

1. **Set up the Python environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Build the Gradle project**:
   ```bash
   cd gradle-project
   ./gradlew build
   cd ..
   ```

## 🛠️ Usage

### 1. Run Full Validation

Run all validation checks:
```bash
./scripts/validate_runtime.sh
```

### 2. Run Demo Validation

Run the demo validation suite:
```bash
./scripts/demo_validation.sh
```

### 3. Run Specific Validations

Run Python-based validations:
```bash
python scripts/validate_tests.py [options]

Options:
  --test-file TEST_FILE    Specific test file to validate
  --report-file REPORT    Output report file (default: reports/validation_report.json)
  --verbose               Enable verbose output
```

## 📊 Understanding the Results

Validation results are saved in `reports/validation_results.json` with the following structure:

```json
{
  "timestamp": "2023-01-01T12:00:00Z",
  "summary": {
    "total_tests": 42,
    "passed": 40,
    "failed": 2,
    "success_rate": 95.24
  },
  "details": [
    {
      "test_name": "CalculatorTest.add",
      "status": "PASSED",
      "execution_time_ms": 42,
      "error_message": null
    }
  ]
}
```

## 🔧 Configuration

Edit `config/validation.conf` to customize validation behavior:

```ini
[validation]
timeout_seconds = 300
max_workers = 4
log_level = INFO

[paths]
test_sources = "src/test/kotlin"
reports_dir = "reports"
```

## 🤝 Contributing

When adding new validations:
1. Add your validation script to `scripts/`
2. Update the documentation in `docs/`
3. Add test cases to `gradle-project/src/test/`
4. Update the main validation script to include your new checks

## 📚 Documentation

For detailed information, see:
- [Runtime Validation Guide](docs/RUNTIME_VALIDATION_GUIDE.md)
- [Adding New Validations](docs/ADDING_VALIDATIONS.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## 🔍 Troubleshooting

Common issues and solutions:

1. **Gradle build failures**:
   - Ensure Java 11+ is installed and set as default
   - Run `./gradlew clean build --info` for detailed error messages

2. **Python dependency issues**:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Test timeouts**:
   - Increase the timeout in `config/validation.conf`
   - Check for long-running test cases

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
