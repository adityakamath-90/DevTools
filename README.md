# DevTools

🛠️ DevTool: KDoc & UnitTest Generator using CodeLlama
DevTool is a Python-based developer utility that leverages the CodeLlama LLM model to automatically generate KDocs (Kotlin documentation) and Unit Test cases for your Kotlin source code. It helps improve developer productivity and ensures consistent, high-quality documentation and testing.

🚀 Features
🔍 Analyze Kotlin source files

📝 Generate rich KDocs for classes, methods, and properties

✅ Generate unit test cases based on method signatures and logic

🤖 Powered by CodeLlama LLM

💻 Cross-platform: Mac, Linux, Windows

📦 Installation
1. Clone this repository
bash
Copy
Edit
git clone https://github.com/your-username/devtool-codellama.git
cd devtool-codellama
2. Create and activate a Python virtual environment (optional but recommended)
bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate  # Linux & macOS
venv\Scripts\activate     # Windows
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
🧠 Running CodeLlama Locally
You will need to run the CodeLlama model locally for this tool to work. Here's how to do it on macOS, Linux, and Windows using llama.cpp or Ollama.

✅ Recommended: Use Ollama (Simple LLM serving)
🔧 Install Ollama
macOS:

bash
Copy
Edit
brew install ollama
Linux:

bash
Copy
Edit
curl -fsSL https://ollama.com/install.sh | sh
Windows:
Download and install from: https://ollama.com/download

📥 Pull CodeLlama Model
bash
Copy
Edit
ollama pull codellama:latest
▶️ Run the model
bash
Copy
Edit
ollama run codellama
This serves the model locally at http://localhost:11434 by default.

🧪 Using DevTool
Once CodeLlama is running, you can use the tool like this:

bash
Copy
Edit
python main.py --input my_kotlin_file.kt --output generated_docs.kt --tests generated_tests.kt
Example Options:
Argument	Description
--input	Path to input Kotlin source file
--output	File to write the generated KDocs
--tests	File to write the generated test cases

⚙️ Configuration
You can configure API endpoints and model parameters in config.yaml (if applicable), or directly through the code.


💡 Troubleshooting
CodeLlama doesn't respond: Ensure it's running locally via Ollama and listening on the expected port (11434).

Model not found: Use ollama pull codellama to ensure the model is downloaded.

Permissions: Use chmod +x if you face script permission issues on macOS/Linux.

🧩 Roadmap
 Add VS Code extension integration

 Support for Java and Swift

 GUI Tool

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss your ideas.

📝 License
This project is licensed under the MIT License. See LICENSE for more information.
