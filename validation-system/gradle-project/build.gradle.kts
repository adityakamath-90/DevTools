plugins {
    kotlin("jvm") version "1.9.10"
    application
}

repositories {
    mavenCentral()
}

dependencies {
    // Kotlin standard library
    implementation("org.jetbrains.kotlin:kotlin-stdlib")
    
    // JUnit 5
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.9.2")
    testImplementation("org.junit.jupiter:junit-jupiter-engine:5.9.2")
    testImplementation("org.junit.jupiter:junit-jupiter-params:5.9.2")
    
    // Kotlin test
    testImplementation("org.jetbrains.kotlin:kotlin-test")
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit5")
    
    // MockK for mocking
    testImplementation("io.mockk:mockk:1.13.4")
    
    // Assertions
    testImplementation("org.assertj:assertj-core:3.24.2")
}

tasks.test {
    useJUnitPlatform()
    
    // Generate test reports
    testLogging {
        events("passed", "skipped", "failed")
        showExceptions = true
        showCauses = true
        showStackTraces = true
    }
    
    // Fail fast on first test failure for quick feedback
    failFast = false
}

// Task to validate generated tests
tasks.register("validateGeneratedTests") {
    dependsOn("test")
    group = "verification"
    description = "Validate all generated Kotlin tests"
    
    doLast {
        val testResultsDir = file("build/test-results/test")
        val testReportsDir = file("build/reports/tests/test")
        
        if (testResultsDir.exists()) {
            println("📊 Test Results Available:")
            println("📁 XML Results: ${testResultsDir.absolutePath}")
            println("📁 HTML Report: ${testReportsDir.absolutePath}/index.html")
            
            // Parse test results
            val xmlFiles = testResultsDir.listFiles { _, name -> name.endsWith(".xml") }
            xmlFiles?.forEach { xmlFile ->
                println("📄 ${xmlFile.name}")
            }
        } else {
            println("❌ No test results found")
        }
    }
}

// Task to run only generated tests
tasks.register<Test>("runGeneratedTests") {
    group = "verification"
    description = "Run only the generated test files"
    
    // Filter to only run generated tests
    filter {
        includeTestsMatching("*Test")
    }
    
    useJUnitPlatform()
    
    testLogging {
        events("passed", "skipped", "failed")
        showStandardStreams = true
    }
}

// Task to check test coverage
tasks.register("testCoverage") {
    dependsOn("test")
    group = "verification"
    description = "Generate test coverage report"
    
    doLast {
        println("🔍 To enable test coverage, add JaCoCo plugin:")
        println("plugins { id(\"jacoco\") }")
    }
}

kotlin {
    jvmToolchain(17)
}
