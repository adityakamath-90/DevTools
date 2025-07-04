#!/bin/bash

echo "🎯 Runtime Validation Demo"
echo "========================="
echo ""

echo "📋 Current Test Files:"
ls -la output-test/
echo ""

echo "🔍 Running Python Static Analysis..."
python validate_tests.py
echo ""

echo "📊 Validation Results Summary:"
if [ -f "validation_results.json" ]; then
    echo "📄 Detailed results available in: validation_results.json"
    
    # Extract key metrics using jq if available, otherwise use basic parsing
    if command -v jq &> /dev/null; then
        echo "📈 Success Rate: $(jq -r '.success_rate' validation_results.json)%"
        echo "✅ Passed: $(jq -r '.passed' validation_results.json)"
        echo "❌ Failed: $(jq -r '.failed' validation_results.json)"
    else
        echo "📈 Install 'jq' for detailed JSON parsing"
    fi
else
    echo "⚠️ No validation results found"
fi

echo ""
echo "🏗️ Available Runtime Validation Options:"
echo "1. Python Static Analysis: python validate_tests.py"
echo "2. Gradle Compilation: cd validation && gradle compileTestKotlin"
echo "3. Full Pipeline: ./validate_runtime.sh"
echo "4. Individual Test Run: cd validation && gradle test"
echo ""
echo "📚 For complete documentation, see: RUNTIME_VALIDATION_GUIDE.md"
