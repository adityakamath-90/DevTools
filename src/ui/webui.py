import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
import uuid
import time
import json
import logging

# Set page config
st.set_page_config(
    page_title="Kotlin Test Generator",
    page_icon="🧪",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Logging setup (local file in this directory: src/ui/webui.log)
# -----------------------------------------------------------------------------
LOG_PATH = Path(__file__).parent / "webui.log"

_logger = logging.getLogger("webui")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fmt = logging.Formatter('%(message)s')  # we'll emit JSON lines ourselves
    fh.setFormatter(fmt)
    _logger.addHandler(fh)
    _logger.propagate = False

def log_event(action: str, data: dict | None = None):
    try:
        record = {
            "ts": time.time(),
            "action": action,
            "session_id": st.session_state.get("session_id"),
            **(data or {}),
        }
        _logger.info(json.dumps(record, ensure_ascii=False))
    except Exception:
        # Avoid breaking the UI due to logging failures
        pass

# Ensure project root is on path and import the application interfaces
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from main import GenAIApplication
from src.agents.test_pipeline import run_pipeline
from src.models.data_models import GenerationRequest


def get_app() -> GenAIApplication:
    """Get or initialize a singleton GenAIApplication stored in session_state."""
    if 'app' not in st.session_state or st.session_state.app is None:
        _t0 = time.perf_counter()
        st.session_state.app = GenAIApplication()
        _elapsed = time.perf_counter() - _t0
        # Log cold start time taken to construct the application and underlying models/resources
        log_event("app_cold_start", {"elapsed_s": round(_elapsed, 3)})
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
        log_event(
            "generate_tests_complete",
            {
                "elapsed_s": round(_elapsed, 3),
                "source_len": len(kotlin_code or ""),
            },
        )

        # Prefer returning the generated test code directly
        test_code = getattr(result, 'test_code', None)
        if test_code and str(test_code).strip():
            log("Successfully generated test code (direct)")
            log_event(
                "generate_tests_result",
                {"status": "success", "test_len": len(str(test_code))},
            )
            return str(test_code)

        # Fallback: try reading the output file if created
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
            if file_content.strip():
                log("Read test code from saved file")
                log_event(
                    "generate_tests_result",
                    {"status": "success_file", "test_len": len(file_content)},
                )
                return file_content

        # If we reach here, no test was produced
        err = getattr(result, 'error_message', 'No test files were generated.')
        log_event(
            "generate_tests_result",
            {"status": "no_output", "error": str(err)[:500]},
        )
        return f"No test files were generated.\nReason: {err}\n\nDebug Output:\n" + "\n".join(debug_output)

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        log(error_msg)
        log_event("generate_tests_exception", {"error": str(e)[:500]})
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
        log_event(
            "improve_tests_complete",
            {
                "elapsed_s": round(_elapsed, 3),
                "source_len": len(original_code or ""),
                "test_len": len(test_code or ""),
                "feedback_len": len(feedback or ""),
            },
        )

        # Accept plain string or LLM-like object with .text
        if isinstance(response, str) and response.strip():
            improved_test = response
        elif hasattr(response, 'text') and getattr(response, 'text'):
            improved_test = response.text
        else:
            improved_test = test_code

        log("Successfully generated improved test code")
        log_event("improve_tests_result", {"status": "success", "improved_len": len(improved_test or "")})
        return improved_test

    except Exception as e:
        error_msg = f"Error in improve_test_with_feedback: {str(e)}"
        log(error_msg)
        log_event("improve_tests_exception", {"error": str(e)[:500]})
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
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        log_event("session_start", {})

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
        log_event("generate_tests_click", {"source_len": len(st.session_state.kotlin_code or "")})
        with st.spinner("Running full pipeline (generate → compile → coverage)..."):
            # 1) Save current Kotlin code to input-src for the pipeline
            input_dir = os.path.join(project_root, "input-src")
            os.makedirs(input_dir, exist_ok=True)
            input_file = os.path.join(input_dir, "WebUIInput.kt")
            with open(input_file, "w", encoding="utf-8") as f:
                f.write(st.session_state.kotlin_code)

            # 2) Run the pipeline with the UI toggle for LangChain
            cov = run_pipeline(
                source_dir=input_dir,
                output_dir=os.path.join(project_root, "output-test"),
                gradle_project_dir=os.path.join(project_root, "validation-system/gradle-project"),
                coverage_threshold=80.0,
                max_iterations=1,
                use_langchain=use_langchain,
            )

            # 3) Read a generated test to display
            out_dir = Path(project_root) / "output-test"
            test_code = ""
            try:
                first_test = next(out_dir.glob("*.kt"), None)
                if first_test and first_test.exists():
                    test_code = first_test.read_text(encoding="utf-8")
            except Exception:
                pass

            st.session_state.test_output = test_code or f"Pipeline finished. Coverage: {cov:.2f}% (no test file found)"
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

        # ------------------------------------------------------------------
        # Post-generation: Two-column layout
        #   - Left: Improve Test Code (regenerate with feedback)
        #   - Right: Share Feedback to Dev Team (rating + comment)
        # ------------------------------------------------------------------
        col_left, col_right = st.columns([0.65, 0.35])

        with col_left:
            st.subheader("Improve Test Code")
            feedback = st.text_area(
                "Improve AI-Generated Tests",
                height=200,
                key="feedback_input",
                placeholder=(
                    "Add Specific Prompt to refine Generated Test Code. What to fix"
                    "or improve: missing edge cases, incorrect assertions, "
                    "setup/teardown, mocking/stubs, coverage gaps, flaky parts, naming, etc."
                    "Example: Use PowerMockito to mock static methods"
                ),
                help="Your feedback is used to regenerate improved test code."
            )

            if st.button("Improve Test Code"):
                if feedback:
                    log_event(
                        "feedback_submit",
                        {
                            "feedback_len": len(feedback or ""),
                            "message": (feedback or "")[:4000],
                        },
                    )
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

        with col_right:
            st.subheader("Share Feedback to Dev Team")
            st.markdown("**How helpful was the generated test code for shipping your production code?**")
            stars = [1, 2, 3, 4, 5]
            star_labels = {
                1: "⭐ – Unusable: Had to write it all myself.",
                2: "⭐⭐ – Got me started on writing test Code. Good tool to add boiler plate code. I can concentrate on correctnes",
                3: "⭐⭐⭐ – Moderate: Needed noticeable edits.",
                4: "⭐⭐⭐⭐ – Good: Minor tweaks only.",
                5: "⭐⭐⭐⭐⭐ – Excellent: Ready out of the box.",
            }

            # Initialize state for rating tracking
            if 'dev_last_rating' not in st.session_state:
                st.session_state.dev_last_rating = None

            rating = st.radio(
                label="Rating",
                options=stars,
                format_func=lambda x: star_labels[x],
                horizontal=False,
                key="dev_rating",
                label_visibility="collapsed",
            )

            # Log rating selection once per change
            if rating and rating != st.session_state.dev_last_rating:
                log_event("rating_select", {"rating": int(rating)})
                st.session_state.dev_last_rating = rating

            # Show feedback box once a rating is chosen
            if rating:
                dev_feedback = st.text_area(
                    "Share Feedback to dev team",
                    height=120,
                    key="dev_feedback_text",
                    placeholder="What worked well? What was missing? Any failures or improvements needed?",
                )

                if st.button("Submit to Dev Team"):
                    log_event(
                        "dev_feedback_submit",
                        {
                            "rating": int(rating),
                            "message_len": len(dev_feedback or ""),
                            "message": (dev_feedback or "")[:4000],
                        },
                    )
                    st.success("Thanks! Your feedback has been recorded for the dev team.")

if __name__ == "__main__":
    main()