from typing import List

from typing import List

class PromptBuilder:
    @staticmethod
    def build_generation_prompt(class_name: str, class_code: str, similar_tests: List[str]) -> str:
        context = "\n\n---\n\n".join(similar_tests) if similar_tests else "// No relevant test cases found in corpus."
        return (
            f"You are a senior Android Kotlin developer tasked with writing comprehensive unit tests for the following Kotlin class: `{class_name}`.\n\n"
            f"Class Code:\n{class_code}\n\n"
            f"You may refer to these similar existing unit tests for context:\n{context}\n\n"
            "Requirements:\n"
            "- Cover all public methods in the class.\n"
            "- Include test cases for:\n"
            "  • Typical usage scenarios\n"
            "  • Edge cases\n"
            "  • Error and exception handling\n"
            "- Use idiomatic Kotlin test style with:\n"
            "  • JUnit 5 for structure\n"
            "  • MockK for mocking any dependencies\n"
            "- Use assertions such as assertEquals, assertTrue, assertFailsWith, etc.\n"
            "- Write meaningful test function names that describe what each test is verifying.\n"
            "- Return only pure Kotlin unit test code.\n"
            "- Do NOT include comments, explanations, markdown, or annotations beyond necessary test-related syntax.\n"
            "\nRespond ONLY with the Kotlin test source code."
        )

    @staticmethod
    def generate_accurate_prompt(class_code: str, generated_test: str) -> str:
        return (
            "You are a senior Android Kotlin developer reviewing a set of proposed unit tests.\n\n"
            f"Here is the Kotlin class being tested:\n{class_code}\n\n"
            f"Here are the proposed unit tests:\n{generated_test}\n\n"
            "Please do the following:\n"
            "1. Confirm that the tests cover all key behaviors, public methods, edge cases, and exception paths.\n"
            "2. Identify any missing tests, logical flaws, or testing anti-patterns.\n"
            "3. Improve or rewrite tests where necessary to ensure full, accurate coverage.\n"
            "4. Use JUnit 5 and MockK idiomatically in Kotlin.\n"
            "\nOutput ONLY the corrected Kotlin unit test source code. Do NOT include explanations, comments, markdown, or any introductory text."
        )

