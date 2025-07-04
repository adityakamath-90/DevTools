#!/bin/bash
# Comprehensive validation script for generated Kotlin tests

set -e

echo "🚀 Starting Kotlin Test Validation Pipeline"
echo "============================================"

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output-test"
VALIDATION_DIR="${PROJECT_ROOT}/validation-system/validation"
REPORTS_DIR="${PROJECT_ROOT}/validation-system/validation-reports"

# Create validation workspace
setup_validation_workspace() {
    echo "📁 Setting up validation workspace..."
    
    mkdir -p "${VALIDATION_DIR}/src/main/kotlin"
    mkdir -p "${VALIDATION_DIR}/src/test/kotlin"
    mkdir -p "${REPORTS_DIR}"
    
    # Copy source files (if they exist)
    if [ -d "input-src" ]; then
        echo "📋 Copying source files..."
        cp -r input-src/* "${VALIDATION_DIR}/src/main/kotlin/" 2>/dev/null || true
    fi
    
    # Copy generated tests
    if [ -d "${OUTPUT_DIR}" ]; then
        echo "📋 Copying generated test files..."
        cp -r "${OUTPUT_DIR}"/* "${VALIDATION_DIR}/src/test/kotlin/" 2>/dev/null || true
    else
        echo "❌ No generated tests found in ${OUTPUT_DIR}"
        exit 1
    fi
}

# Validate with Python script
validate_with_python() {
    echo "🐍 Running Python validation..."
    
    if [ -f "validate_tests.py" ]; then
        python validate_tests.py
        echo "✅ Python validation complete"
    else
        echo "⚠️  Python validation script not found"
    fi
}

# Validate with Gradle
validate_with_gradle() {
    echo "🏗️  Running Gradle validation..."
    cd "${VALIDATION_DIR}"

    # Ensure Gradle project exists in validation dir
    if [ ! -f "settings.gradle" ] && [ ! -f "settings.gradle.kts" ]; then
        echo "⚠️  No Gradle project found in ${VALIDATION_DIR}. Initializing new Gradle project..."
        gradle init --type kotlin-application --dsl kotlin --project-name validation || {
            echo "❌ Failed to initialize Gradle project"
            cd "${PROJECT_ROOT}"
            return 1
        }
    fi

    # Check if gradle wrapper exists, if not create a simple gradle command
    if [ -f "gradlew" ]; then
        GRADLE_CMD="./gradlew"
    else
        GRADLE_CMD="gradle"
    fi

    # Check Gradle installation
    if ! command -v ${GRADLE_CMD} &> /dev/null; then
        echo "⚠️  Gradle not found. Skipping Gradle validation."
        cd "${PROJECT_ROOT}"
        return
    fi

    echo "🔍 Compiling tests..."
    ${GRADLE_CMD} compileTestKotlin || {
        echo "❌ Test compilation failed"
        cd "${PROJECT_ROOT}"
        return 1
    }

    echo "🧪 Running tests..."
    ${GRADLE_CMD} test --continue || {
        echo "⚠️  Some tests failed, but continuing..."
    }

    echo "📊 Generating reports..."
    # Removed: ${GRADLE_CMD} validateGeneratedTests || true

    # Copy reports to main reports directory
    if [ -d "build/reports" ]; then
        cp -r build/reports/* "${REPORTS_DIR}/"
        echo "📁 Reports copied to: ${REPORTS_DIR}"
    fi

    cd "${PROJECT_ROOT}"
}

# Static analysis with ktlint (if available)
validate_with_ktlint() {
    echo "🔍 Running static analysis..."
    
    if command -v ktlint &> /dev/null; then
        echo "📋 Running ktlint on generated tests..."
        ktlint "${OUTPUT_DIR}/**/*.kt" || {
            echo "⚠️  Code style issues found"
        }
    else
        echo "⚠️  ktlint not found. Install with: brew install ktlint"
    fi
}

# Validate test file structure
validate_structure() {
    echo "🏗️  Validating test file structure..."
    
    local test_files=($(find "${OUTPUT_DIR}" -name "*.kt" 2>/dev/null))
    local total_files=${#test_files[@]}
    local valid_files=0
    
    if [ ${total_files} -eq 0 ]; then
        echo "❌ No test files found"
        return 1
    fi
    
    echo "📄 Found ${total_files} test files:"
    
    for file in "${test_files[@]}"; do
        local filename=$(basename "${file}")
        echo -n "  📝 ${filename}: "
        
        # Check if file follows naming convention
        if [[ "${filename}" =~ Test\.kt$ ]]; then
            echo -n "✅ naming "
        else
            echo -n "❌ naming "
        fi
        
        # Check if file contains @Test annotations
        if grep -q "@Test" "${file}"; then
            echo -n "✅ tests "
            ((valid_files++))
        else
            echo -n "❌ tests "
        fi
        
        # Check if file has proper class structure
        if grep -q "class.*Test" "${file}"; then
            echo "✅ structure"
        else
            echo "❌ structure"
        fi
    done
    
    echo ""
    echo "📊 Structure validation: ${valid_files}/${total_files} files valid"
}

# Generate final report
generate_report() {
    echo "📊 Generating final validation report..."
    
    local report_file="${REPORTS_DIR}/validation_summary.md"
    
    cat > "${report_file}" << EOF
# Kotlin Test Validation Report

Generated on: $(date)

## Summary

- **Test Files Generated**: $(find "${OUTPUT_DIR}" -name "*.kt" 2>/dev/null | wc -l)
- **Validation Date**: $(date)
- **Project**: AI-Powered Kotlin Test Generation

## Files Validated

EOF

    # List all test files
    find "${OUTPUT_DIR}" -name "*.kt" 2>/dev/null | while read -r file; do
        echo "- \`$(basename "${file}")\`" >> "${report_file}"
    done
    
    cat >> "${report_file}" << EOF

## Validation Methods Used

1. **Python Script Validation**: Syntax and structure checks
2. **Gradle Compilation**: Kotlin compiler validation
3. **Static Analysis**: Code style and best practices
4. **Structure Validation**: File naming and test annotations

## Reports Generated

- **HTML Test Report**: \`${REPORTS_DIR}/tests/test/index.html\`
- **XML Test Results**: \`${REPORTS_DIR}/test-results/\`
- **Validation JSON**: \`validation_results.json\`

## Next Steps

1. Review any failed tests in the HTML report
2. Check validation_results.json for detailed analysis
3. Run tests in your IDE for interactive debugging
4. Integrate with CI/CD pipeline for automated validation

EOF

    echo "📁 Report generated: ${report_file}"
}

# Main execution
main() {
    echo "🎯 Validation Target: ${OUTPUT_DIR}"
    echo ""
    
    # Run all validation steps
    setup_validation_workspace
    validate_structure
    validate_with_python
    validate_with_ktlint
    validate_with_gradle
    generate_report
    
    echo ""
    echo "🎉 Validation pipeline complete!"
    echo "📊 Check reports in: ${REPORTS_DIR}"
    echo ""
    
    # Show quick summary
    if [ -f "validation_results.json" ]; then
        if command -v jq &> /dev/null; then
            echo "📈 Quick Summary:"
            jq -r '"Success Rate: " + (.success_rate | tostring) + "%"' validation_results.json
            jq -r '"Total Files: " + (.total_files | tostring)' validation_results.json
            jq -r '"Passed: " + (.passed | tostring)' validation_results.json
            jq -r '"Failed: " + (.failed | tostring)' validation_results.json
        else
            echo "💡 Install jq for JSON parsing: brew install jq"
        fi
    fi
}

# Run main function
main "$@"
