from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class KotlinVisibility(Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"
    INTERNAL = "internal"

@dataclass
class KotlinParameter:
    name: str
    type: str
    default_value: Optional[str] = None
    is_vararg: bool = False

@dataclass
class KotlinProperty:
    name: str
    type: Optional[str] = None
    visibility: KotlinVisibility = KotlinVisibility.PUBLIC
    is_mutable: bool = True
    is_override: bool = False
    default_value: Optional[str] = None
    getter_body: Optional[str] = None
    setter_body: Optional[str] = None

@dataclass
class KotlinMethod:
    name: str
    parameters: List[KotlinParameter] = field(default_factory=list)
    return_type: Optional[str] = None
    visibility: KotlinVisibility = KotlinVisibility.PUBLIC
    is_override: bool = False
    is_abstract: bool = False
    is_open: bool = False
    is_suspend: bool = False
    body: Optional[str] = None

@dataclass
class KotlinClass:
    name: str
    properties: List[KotlinProperty] = field(default_factory=list)
    methods: List[KotlinMethod] = field(default_factory=list)
    visibility: KotlinVisibility = KotlinVisibility.PUBLIC
    is_abstract: bool = False
    is_open: bool = False
    is_data: bool = False
    parent_class: Optional[str] = None
    interfaces: List[str] = field(default_factory=list)
    source_code: Optional[str] = None
    file_path: Optional[str] = None

@dataclass
class KotlinFile:
    package_name: Optional[str]
    imports: List[str] = field(default_factory=list)
    classes: List[KotlinClass] = field(default_factory=list)
    top_level_functions: List[KotlinMethod] = field(default_factory=list)


# --- GenerationRequest and GenerationResult for LLM/test generation ---
from enum import Enum

class GenerationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class GenerationRequest:
    request_id: str
    class_name: str
    source_code: str
    parameters: Optional[dict] = None
    status: GenerationStatus = GenerationStatus.PENDING
    created_at: Optional[str] = None

@dataclass
class GenerationResult:
    request_id: str
    status: GenerationStatus
    output_file: Optional[str] = None
    test_code: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class SimilarityMatch:
    test_name: str
    similarity_score: float
    matched_code: Optional[str] = None
    source_file: Optional[str] = None

@dataclass
class EmbeddingVector:
    vector: List[float]
    source: Optional[str] = None
    metadata: Optional[dict] = None

@dataclass
class ModelMetrics:
    model_name: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    additional_info: Optional[dict] = None

@dataclass
class TestCase:
    name: str
    input_data: Optional[dict] = None
    expected_output: Optional[any] = None
    description: Optional[str] = None
    passed: Optional[bool] = None
    error_message: Optional[str] = None


# --- Additional common data models for project-wide use ---

@dataclass
class LLMProviderInfo:
    name: str
    version: Optional[str] = None
    endpoint: Optional[str] = None
    api_key: Optional[str] = None  # Should be loaded from env/config, not hardcoded
    description: Optional[str] = None

@dataclass
class EmbeddingRequest:
    input_text: str
    model: Optional[str] = None
    parameters: Optional[dict] = None

@dataclass
class EmbeddingResult:
    vector: List[float]
    model: Optional[str] = None
    input_text: Optional[str] = None
    metadata: Optional[dict] = None

@dataclass
class TestGenerationSummary:
    total_tests: int
    passed: int
    failed: int
    skipped: int = 0
    details: Optional[List[TestCase]] = None

@dataclass
class ErrorInfo:
    error_type: str
    message: str
    traceback: Optional[str] = None

@dataclass
class PromptTemplate:
    name: str
    template: str
    variables: List[str]
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# Example default templates (can be customized as needed)
DEFAULT_GENERATION_TEMPLATE = PromptTemplate(
    name="default_generation",
    template=(
        "You are a senior Android Kotlin developer tasked with writing comprehensive unit tests for the following Kotlin class: `{class_name}`.\n\n"
        "Class Code:\n{class_code}\n\n"
        "You may refer to these similar existing unit tests for context:\n{similar_tests}\n\n"
        "Methods to test:\n{methods_info}\n\n"
        "Requirements:\n"
        "- Cover all public methods in the class.\n"
        "- Include test cases for typical usage scenarios, edge cases, and error handling.\n"
        "- Use idiomatic Kotlin test style with JUnit 5 and MockK.\n"
        "- Use assertions such as assertEquals, assertTrue, assertFailsWith, etc.\n"
        "- Write meaningful test function names that describe what each test is verifying.\n"
        "- Return only pure Kotlin unit test code.\n"
        "- Do NOT include comments, explanations, markdown, or annotations beyond necessary test-related syntax.\n"
        "\nRespond ONLY with the Kotlin test source code."
    ),
    variables=["class_name", "class_code", "similar_tests", "methods_info"],
    description="Default template for generating Kotlin tests"
)

DEFAULT_ACCURACY_TEMPLATE = PromptTemplate(
    name="default_accuracy",
    template=(
        "Check if the following test covers all public methods of the class:\nClass:\n{class_code}\nTest:\n{test_code}\n"
    ),
    variables=["class_code", "test_code"],
    description="Default template for test coverage accuracy"
)