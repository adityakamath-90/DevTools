# app.py
import streamlit as st
import os
import sys
from pathlib import Path
import subprocess
import tempfile

# Set page config
st.set_page_config(
    page_title="Kotlin Test Generator",
    page_icon="🧪",
    layout="wide"
)

def run_test_generation(kotlin_code: str) -> str:
    """Run the test generation with the provided Kotlin code"""
    debug_output = []
    
    def log(msg):
        debug_output.append(msg)
        print(msg)  # Also print to console for debugging
    
    log("Starting test generation...")
    
    try:
        # Get the absolute path to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        main_script = os.path.join(project_root, 'main.py')
        log(f"Project root: {project_root}")
        log(f"Main script: {main_script}")
        
        if not os.path.exists(main_script):
            error_msg = f"Error: Could not find main.py at {main_script}"
            log(error_msg)
            return error_msg
        
        # Create a temporary directory for the input file
        with tempfile.TemporaryDirectory() as temp_dir:
            log(f"Created temp directory: {temp_dir}")
            
            # Create the input directory structure that main.py expects
            input_src_dir = os.path.join(temp_dir, 'input-src')
            os.makedirs(input_src_dir, exist_ok=True)
            log(f"Created input directory: {input_src_dir}")
            
            # Create the Kotlin file in the input directory
            input_file = os.path.join(input_src_dir, 'Input.kt')
            log(f"Writing Kotlin code to: {input_file}")
            with open(input_file, 'w', encoding='utf-8') as f:
                f.write(kotlin_code)
            
            # Create output directory
            output_dir = os.path.join(temp_dir, 'output-test')
            os.makedirs(output_dir, exist_ok=True)
            log(f"Created output directory: {output_dir}")
            
            # Prepare the command
            cmd = [
                sys.executable, 
                main_script, 
                'test',
                '--source-dir', input_src_dir,
                '--output-dir', output_dir,
                '--debug'  # Enable debug output
            ]
            log(f"Running command: {' '.join(cmd)}")
            
            try:
                # Run the test generation with the correct arguments
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=project_root,
                    timeout=300  # 5 minute timeout
                )
                log("Command execution completed")
                log(f"Return code: {result.returncode}")
                
                # Log the output for debugging
                if result.stdout:
                    log("=== STDOUT ===")
                    log(result.stdout)
                if result.stderr:
                    log("=== STDERR ===")
                    log(result.stderr)
                
                # Check for generated test files
                generated_tests = list(Path(output_dir).rglob('*Test.kt'))
                log(f"Found {len(generated_tests)} generated test files")
                
                if generated_tests:
                    # Return the first test file found
                    test_file = generated_tests[0]
                    log(f"Reading test file: {test_file}")
                    with open(test_file, 'r', encoding='utf-8') as f:
                        test_content = f.read()
                    log("Successfully read test file content")
                    return test_content
                
                # If no test files found, return the command output
                output = result.stdout or "No output from test generation"
                if result.stderr:
                    output = f"{output}\n\nError Output:\n{result.stderr}"
                
                return f"No test files were generated.\n\nDebug Output:\n" + "\n".join(debug_output)
                
            except subprocess.TimeoutExpired:
                error_msg = "Test generation timed out after 5 minutes"
                log(error_msg)
                return f"{error_msg}\n\nDebug Output:\n" + "\n".join(debug_output)
                
            except Exception as e:
                error_msg = f"Error during test generation: {str(e)}"
                log(error_msg)
                return f"{error_msg}\n\nDebug Output:\n" + "\n".join(debug_output)
                
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        log(error_msg)
        return f"{error_msg}\n\nDebug Output:\n" + "\n".join(debug_output)

def main():
    st.title("🧪 Kotlin Test Generator")
    st.markdown("Upload a Kotlin file or paste the code to generate unit tests")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Kotlin file", type=["kt"])
    
    # Text area for direct code input
    code_input = st.text_area("Or paste Kotlin code here", height=300)
    
    kotlin_code = ""
    if uploaded_file is not None:
        kotlin_code = uploaded_file.getvalue().decode("utf-8")
    elif code_input:
        kotlin_code = code_input
    
    # Display the input code
    if kotlin_code:
        with st.expander("Input Code", expanded=True):
            st.code(kotlin_code, language="kotlin")
    
    # Generate tests button
    if st.button("Generate Tests", type="primary") and kotlin_code:
        with st.spinner("Generating tests..."):
            test_output = run_test_generation(kotlin_code)
            
            # Display test output
            st.subheader("Generated Test Code")
            st.code(test_output, language="kotlin")
            
            # Download button for the test file
            st.download_button(
                label="Download Test File",
                data=test_output,
                file_name="GeneratedTest.kt",
                mime="text/x-kotlin"
            )
            
            # Feedback section
            st.subheader("Feedback")
            feedback = st.text_area("Provide feedback to improve the test generation", height=100)
            
            if st.button("Submit Feedback"):
                if feedback:
                    # Here you would typically send the feedback to your feedback system
                    st.success("Thank you for your feedback! We'll use it to improve the test generation.")
                    # TODO: Implement feedback submission logic
                else:
                    st.warning("Please provide some feedback before submitting")

if __name__ == "__main__":
    main()