import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
import uuid
import time

# Set page config
st.set_page_config(
    page_title="Kotlin Test Generator",
    page_icon="🧪",
    layout="wide"
)

# Ensure project root is on path and import the application interfaces
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from main import GenAIApplication
from src.models.data_models import GenerationRequest


def get_app() -> GenAIApplication:
    """Get or initialize a singleton GenAIApplication stored in session_state."""
    if 'app' not in st.session_state or st.session_state.app is None:
        st.session_state.app = GenAIApplication()
    return st.session_state.app


def run_test_generation(kotlin_code: str, use_langchain: bool = True) -> str:
    """Run test generation directly via GenAIApplication without subprocesses."""
    debug_output = []

    def log(msg):
        debug_output.append(msg)
        print(msg)

    log("Starting test generation (direct)...")

    try:
        app = get_app()
        # Derive a class name from the Kotlin code
        class_name = app.test_generator.parser.extract_class_name(kotlin_code) or "Input"

        # Prepare output path in a temp dir (so files are available if needed)
        out_dir = tempfile.mkdtemp(prefix='output-test-')
        output_file = os.path.join(out_dir, f"{class_name}Test.kt")

        # Build a GenerationRequest
        request = GenerationRequest(
            request_id=str(uuid.uuid4()),
            class_name=class_name,
            source_code=kotlin_code,
            parameters=None,
        )
        setattr(request, 'output_file', output_file)

        # Generate tests
        log(f"Generating tests for class: {class_name}")
        _t0 = time.perf_counter()
        result = app.test_generator.generate_tests(request)
        _elapsed = time.perf_counter() - _t0
        log(f"LLM/test generation took {_elapsed:.2f}s")

        # Prefer returning the generated test code directly
        test_code = getattr(result, 'test_code', None)
        if test_code and str(test_code).strip():
            log("Successfully generated test code (direct)")
            return str(test_code)

        # Fallback: try reading the output file if created
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
            if file_content.strip():
                log("Read test code from saved file")
                return file_content

        # If we reach here, no test was produced
        err = getattr(result, 'error_message', 'No test files were generated.')
        return f"No test files were generated.\nReason: {err}\n\nDebug Output:\n" + "\n".join(debug_output)

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        log(error_msg)
        return f"{error_msg}\n\nDebug Output:\n" + "\n".join(debug_output)


def improve_test_with_feedback(original_code: str, test_code: str, feedback: str) -> str:
    """
    Generate improved test code using the provided feedback by leveraging the GenAIApplication.

    Args:
        original_code: The original Kotlin source code
        test_code: The generated test code
        feedback: User feedback for improvement

    Returns:
        Improved test code
    """
    debug_output = []

    def log(msg):
        debug_output.append(msg)
        print(msg)  # Also print to console for debugging

    log("Starting test improvement with feedback...")
    log(f"Feedback received: {feedback}")

    try:
        # Reuse the persisted application
        app = get_app()

        # Use the application's method to improve tests with feedback
        log("Generating improved tests with feedback...")
        _t0 = time.perf_counter()
        response = app.improve_tests_with_feedback(
            source_code=original_code,
            generated_test_code=test_code,
            user_feedback=feedback,
            output_dir=os.path.join(tempfile.gettempdir(), 'improved_tests')
        )
        _elapsed = time.perf_counter() - _t0
        log(f"Test improvement (feedback) took {_elapsed:.2f}s")

        # Accept plain string or LLM-like object with .text
        if isinstance(response, str) and response.strip():
            improved_test = response
        elif hasattr(response, 'text') and getattr(response, 'text'):
            improved_test = response.text
        else:
            improved_test = test_code

        log("Successfully generated improved test code")
        return improved_test

    except Exception as e:
        error_msg = f"Error in improve_test_with_feedback: {str(e)}"
        log(error_msg)
        return test_code  # Return original if error occurs


def main():
    st.title("🧪 Kotlin Test Generator")
    st.markdown("Upload a Kotlin file or paste the code to generate unit tests")

    # Initialize session state variables
    if 'kotlin_code' not in st.session_state:
        st.session_state.kotlin_code = ""
    if 'test_output' not in st.session_state:
        st.session_state.test_output = ""
    if 'show_feedback' not in st.session_state:
        st.session_state.show_feedback = False

    # File uploader
    uploaded_file = st.file_uploader("Upload Kotlin file", type=["kt"])

    # Text area for direct code input
    code_input = st.text_area("Or paste Kotlin code here", height=300, key="code_input")

    # Get the Kotlin code
    if uploaded_file is not None:
        st.session_state.kotlin_code = uploaded_file.getvalue().decode("utf-8")
    elif code_input:
        st.session_state.kotlin_code = code_input

    # Display the input code
    if st.session_state.kotlin_code:
        with st.expander("Input Code", expanded=True):
            st.code(st.session_state.kotlin_code, language="kotlin")

    # Options (LangChain toggle retained for future pipeline runs; ignored in direct mode)
    use_langchain = st.checkbox("Use LangChain orchestrator", value=True)

    # Ensure app is initialized early to avoid cold start on button click
    _ = get_app()

    # Generate tests button
    if st.button("Generate Tests", type="primary") and st.session_state.kotlin_code:
        with st.spinner("Generating tests..."):
            st.session_state.test_output = run_test_generation(st.session_state.kotlin_code, use_langchain)
            st.session_state.show_feedback = True
            st.rerun()

    # Display test output if available
    if st.session_state.test_output:
        st.subheader("Generated Test Code")
        st.code(st.session_state.test_output, language="kotlin")

        # Download button for the test file
        st.download_button(
            label="Download Test File",
            data=st.session_state.test_output,
            file_name="GeneratedTest.kt",
            mime="text/x-kotlin"
        )

        # Feedback section
        if st.session_state.show_feedback:
            st.subheader("Feedback")
            feedback = st.text_area(
                "Provide feedback to improve the test generation",
                height=100,
                key="feedback_input"
            )

            if st.button("Submit Feedback"):
                if feedback:
                    with st.spinner("Improving tests based on your feedback..."):
                        # Generate improved test code using the feedback
                        improved_test = improve_test_with_feedback(
                            st.session_state.kotlin_code,
                            st.session_state.test_output,
                            feedback
                        )

                        # Update the test output with the improved version
                        st.session_state.test_output = improved_test
                        st.session_state.show_feedback = True  # Keep feedback visible for iterative improvements
                        st.success("Test code has been improved based on your feedback!")
                        st.rerun()
                else:
                    st.warning("Please provide some feedback before submitting")

if __name__ == "__main__":
    main()