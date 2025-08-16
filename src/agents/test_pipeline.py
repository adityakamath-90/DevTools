"""
Multi-agent pipeline to read Kotlin source, generate tests using local Ollama (CodeLlama),
compile and run tests via Gradle, and loop until coverage >= threshold using JaCoCo.

Agents:
- SourceReaderAgent: reads Kotlin source from config.directories.input_dir
- TestGenerationAgent: uses existing TestGeneratorAgent (local Ollama)
- CompileAgent: syncs files into validation-system Gradle project and runs tests
- CoverageAgent: parses JaCoCo XML and reports overall coverage
 - StaticAnalysisAgent: improves generated tests using static analysis

Everything is local-only.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from config.settings import GenerationConfig, EmbeddingConfig
from utils.logging import get_logger
from services.llm_agent import create_test_generator_agent
from services.embedding_service import EmbeddingIndexerService, SimpleEmbeddingIndexerService
from core.code_parser import KotlinParser
from models.data_models import KotlinClass

# Optional: LangChain runnables for orchestrating steps
try:
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    LANGCHAIN_RUNNABLES_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    LANGCHAIN_RUNNABLES_AVAILABLE = False

# Safe imports for LangChain-enabled path
try:
    # Preferred import paths (when 'src' is on PYTHONPATH)
    from services.llm_service import LLMService  # type: ignore
except Exception:  # pragma: no cover - fallback if package layout differs
    try:
        from src.services.llm_service import LLMService  # type: ignore
    except Exception:  # Final fallback: disable LangChain path
        LLMService = None  # type: ignore

try:
    from core.test_generator import KotlinTestGenerator  # type: ignore
except Exception:  # pragma: no cover
    try:
        from src.core.test_generator import KotlinTestGenerator  # type: ignore
    except Exception:
        KotlinTestGenerator = None  # type: ignore

LOGGER = get_logger("agents.pipeline")


# -------- Agent 1: read_source_code --------
class SourceReaderAgent:
    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)
        self.logger = get_logger(self.__class__.__name__)

    def read(self) -> List[Tuple[Path, str]]:
        self.logger.info(f"Scanning for Kotlin files in: {self.source_dir}")
        files: List[Tuple[Path, str]] = []
        if not self.source_dir.exists():
            self.logger.warning(f"Source dir does not exist: {self.source_dir}")
            return files
        for p in self.source_dir.rglob("*.kt"):
            # Exclude test files
            if "test" in p.name.lower():
                continue
            try:
                text = p.read_text(encoding="utf-8")
                files.append((p, text))
            except Exception as e:
                self.logger.error(f"Failed reading {p}: {e}")
        self.logger.info(f"Found {len(files)} Kotlin source files")
        return files


# -------- Agent 2: generate_tests --------
class TestGenerationAgent:
    def __init__(self, output_dir: str, reference_tests_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reference_tests_dir = Path(reference_tests_dir)
        self.logger = get_logger(self.__class__.__name__)

        # Prefer advanced embedding indexer (CodeBERT + FAISS), fallback to simple
        try:
            emb_cfg = EmbeddingConfig(test_cases_dir=str(self.reference_tests_dir))
            self.sim_index = EmbeddingIndexerService(emb_cfg)
            self.logger.info("Using advanced EmbeddingIndexerService for similarity search")
        except Exception as e:
            self.logger.warning(f"Advanced embedding service failed: {e}; using simple indexer")
            self.sim_index = SimpleEmbeddingIndexerService(str(self.reference_tests_dir))

        # Prefer core KotlinTestGenerator with LLMService (which can use LangChain)
        self.use_core_generator = False
        self.core_generator = None
        if LLMService is not None and KotlinTestGenerator is not None:
            try:
                llm_service = LLMService()  # Reads config to choose provider (LangChain if enabled)
                gen_cfg = GenerationConfig(output_dir=str(self.output_dir))
                self.core_generator = KotlinTestGenerator(llm_service, self.sim_index, gen_cfg)
                self.use_core_generator = True
                self.logger.info("Using core KotlinTestGenerator with LLMService (LangChain-capable)")
            except Exception as e:
                self.logger.warning(f"Failed to init core generator; falling back to simple agent: {e}")
                self.use_core_generator = False

        # Fallback: LLM-based generator using local Ollama via existing simple agent
        if not self.use_core_generator:
            self.generator = create_test_generator_agent()

    def _class_name_from_source(self, source_code: str, default: str = "Generated") -> str:
        # Prefer robust parsing via KotlinParser
        try:
            parser = KotlinParser()
            name = parser.extract_class_name(source_code)
            if name:
                return name
        except Exception:
            pass
        # Fallback to simple regex if parser fails
        m = re.search(r"class\s+(\w+)", source_code)
        return m.group(1) if m else default

    def generate_for_sources(self, items: List[Tuple[Path, str]]) -> List[Path]:
        generated: List[Path] = []
        for file_path, source_code in items:
            try:
                class_name = self._class_name_from_source(source_code, default=file_path.stem)
                test_class_name = f"{class_name}Test"
                if self.use_core_generator and self.core_generator is not None:
                    # Use the robust core generator; it will save to output_dir itself
                    from models.data_models import GenerationRequest  # local import to avoid import cycles
                    req = GenerationRequest(
                        request_id=f"req::{file_path.stem}",
                        class_name=class_name,
                        source_code=source_code,
                        parameters=None,
                    )
                    setattr(req, 'output_file', str(self.output_dir / f"{test_class_name}.kt"))
                    result = self.core_generator.generate_tests(req)
                    if getattr(result, 'output_file', None):
                        generated.append(Path(result.output_file))
                        self.logger.info(f"Generated test (core): {result.output_file}")
                    else:
                        self.logger.warning(f"Core generator produced no output for {file_path.name}")
                else:
                    # Fallback to simple agent
                    similar = []
                    try:
                        similar = self.sim_index.find_similar(source_code, top_k=3)
                    except Exception as e:
                        self.logger.warning(f"Similarity lookup failed for {file_path.name}: {e}")

                    test_code = self.generator.generate_test(
                        code=source_code,
                        test_class_name=test_class_name,
                        similar_tests=similar or None,
                    )

                    # Extract code blocks if present
                    blocks = self.generator.llm_agent.extract_code_blocks(test_code)
                    if blocks:
                        test_code = "\n\n".join(cb.strip() for cb in blocks)

                    # Basic success heuristic
                    if not any(tok in test_code for tok in ("@Test", "class ", "fun ")):
                        self.logger.warning(f"Heuristic failed for {file_path.name}; saving raw output anyway")

                    out_name = f"{test_class_name}.kt"
                    out_path = self.output_dir / out_name
                    out_path.write_text(test_code, encoding="utf-8")
                    generated.append(out_path)
                    self.logger.info(f"Generated test: {out_path}")
            except Exception as e:
                self.logger.error(f"Generation failed for {file_path}: {e}")
        return generated


# -------- Agent 2.5: improve_tests (Static analysis)
class StaticAnalysisAgent:
    def __init__(self, generator_agent: 'TestGenerationAgent'):
        self.logger = get_logger(self.__class__.__name__)
        self.generator_agent = generator_agent
        # Prefer reusing the core KotlinTestGenerator if available
        self.core_generator = None
        if getattr(generator_agent, 'use_core_generator', False) and generator_agent.core_generator is not None:
            try:
                # Clone config with validation + static analysis enabled
                gen_cfg = GenerationConfig(
                    output_dir=str(generator_agent.output_dir),
                    enable_validation=True,
                    enable_static_analysis=True,
                    enable_compilation_checks=True,  # required to reach static analysis block
                    enable_auto_fix=False,
                    enable_coverage_checks=False,
                    enable_coverage_improvement=False,
                )
                # Reuse same LLMService and SimilarityIndexer by constructing a new generator with same deps
                llm_service = generator_agent.core_generator.llm_provider
                sim_index = generator_agent.sim_index
                self.core_generator = KotlinTestGenerator(llm_service, sim_index, gen_cfg)
                self.logger.info("StaticAnalysisAgent initialized with core KotlinTestGenerator")
            except Exception as e:
                self.logger.warning(f"StaticAnalysisAgent failed to initialize core generator: {e}")
                self.core_generator = None
        else:
            self.logger.info("StaticAnalysisAgent running in no-op mode (core generator unavailable)")

    def improve_dir(self, sources: List[Tuple[Path, str]], tests_dir: str) -> int:
        """Improve tests in tests_dir using static analysis for provided sources.
        Returns number of files improved.
        """
        if self.core_generator is None:
            return 0
        improved_count = 0
        # Build a map from class name to KotlinClass for quick lookup
        parser = KotlinParser()
        class_map = {}
        for src_path, src_code in sources:
            try:
                name = parser.extract_class_name(src_code)
                if name:
                    class_map[name] = KotlinClass(name=name, source_code=src_code)
            except Exception:
                continue
        for test_path in Path(tests_dir).glob("*.kt"):
            try:
                code = test_path.read_text(encoding="utf-8")
                # Heuristic: test file <ClassName>Test.kt
                base = test_path.stem
                cls = None
                if base.endswith("Test"):
                    orig = base[:-4]
                    cls = class_map.get(orig)
                # Fallback: if not found, try any class (still allows generic cleanups)
                if cls is None and class_map:
                    cls = next(iter(class_map.values()))
                if cls is None:
                    continue
                new_code = self.core_generator._validate_and_improve_test(cls, code)
                if new_code and new_code != code:
                    test_path.write_text(new_code, encoding="utf-8")
                    improved_count += 1
                    self.logger.info(f"Improved test: {test_path.name}")
            except Exception as e:
                self.logger.warning(f"Failed improving {test_path.name}: {e}")
        return improved_count

# -------- Agent 3: compile_tests (Gradle project) --------
class CompileAgent:
    def __init__(self, gradle_project_dir: str):
        self.gradle_dir = Path(gradle_project_dir)
        self.src_main = self.gradle_dir / "src/main/kotlin"
        self.src_test = self.gradle_dir / "src/test/kotlin"
        self.logger = get_logger(self.__class__.__name__)
        self.src_main.mkdir(parents=True, exist_ok=True)
        self.src_test.mkdir(parents=True, exist_ok=True)

    def sync_sources(self, kotlin_sources: List[Tuple[Path, str]]) -> int:
        count = 0
        for p, content in kotlin_sources:
            dest = self.src_main / p.name
            dest.write_text(content, encoding="utf-8")
            count += 1
        self.logger.info(f"Synced {count} sources -> {self.src_main}")
        return count

    def sync_tests_from_dir(self, tests_dir: str) -> int:
        count = 0
        for p in Path(tests_dir).glob("*.kt"):
            shutil.copy2(p, self.src_test / p.name)
            count += 1
        self.logger.info(f"Synced {count} tests -> {self.src_test}")
        return count

    def _gradle_cmd(self) -> List[str]:
        wrapper = self.gradle_dir / "gradlew"
        if wrapper.exists():
            return [str(wrapper)]
        return ["gradle"]

    def run_tests(self) -> bool:
        cmd_base = self._gradle_cmd()
        cmds = [
            cmd_base + ["test"],
            cmd_base + ["jacocoTestReport"],
        ]
        ok = True
        for cmd in cmds:
            self.logger.info(f"Running: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, cwd=str(self.gradle_dir), check=True)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Gradle command failed: {e}")
                ok = False
        return ok


# -------- Agent 4: check_coverage (JaCoCo) --------
class CoverageAgent:
    def __init__(self, gradle_project_dir: str):
        self.gradle_dir = Path(gradle_project_dir)
        self.report_xml = self.gradle_dir / "build/reports/jacoco/test/jacocoTestReport.xml"
        self.logger = get_logger(self.__class__.__name__)

    def get_coverage(self) -> float:
        if not self.report_xml.exists():
            self.logger.warning(f"Coverage report not found: {self.report_xml}")
            return 0.0
        try:
            tree = ET.parse(self.report_xml)
            root = tree.getroot()
            # Compute instruction coverage: covered / (covered + missed)
            covered = 0
            missed = 0
            for counter in root.iter("counter"):
                if counter.attrib.get("type") == "INSTRUCTION":
                    covered += int(counter.attrib.get("covered", 0))
                    missed += int(counter.attrib.get("missed", 0))
            total = covered + missed
            return (covered / total) * 100.0 if total > 0 else 0.0
        except Exception as e:
            self.logger.error(f"Failed to parse JaCoCo XML: {e}")
            return 0.0


# -------- Orchestrator --------
@dataclass
class PipelineConfig:
    source_dir: str = "input-src"
    output_dir: str = "output-test"
    gradle_project_dir: str = "validation-system/gradle-project"
    coverage_threshold: float = 80.0
    max_iterations: int = 1
    use_langchain: bool = True  


class MultiAgentOrchestrator:
    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        self.reader = SourceReaderAgent(self.cfg.source_dir)
        self.generator = TestGenerationAgent(self.cfg.output_dir, reference_tests_dir="testcase-datastore")
        self.compiler = CompileAgent(self.cfg.gradle_project_dir)
        self.coverage = CoverageAgent(self.cfg.gradle_project_dir)
        self.logger = get_logger(self.__class__.__name__)
        self.improver = StaticAnalysisAgent(self.generator)

    def _run_with_langchain(self) -> float:
        """Run the pipeline using LangChain runnables to model the iteration steps."""
        if not LANGCHAIN_RUNNABLES_AVAILABLE:
            self.logger.warning("LangChain not available; falling back to built-in loop")
            return self._run_builtin()

        # Step 1: read
        sources = self.reader.read()
        if not sources:
            self.logger.error("No Kotlin sources found to generate tests for.")
            return 0.0

        # Initial sync of sources into Gradle project (outside the iteration chain)
        self.compiler.sync_sources(sources)

        # Define a per-iteration chain using runnables
        # Context schema: {"sources": List[Tuple[Path, str]], "tests": Optional[List[Path]], "improved": int, "ok": bool, "coverage": float}
        iteration_chain = (
            RunnablePassthrough()
            .assign(
                tests=lambda ctx: self.generator.generate_for_sources(ctx["sources"])  # generate tests
            )
            .assign(
                improved=lambda ctx: self.improver.improve_dir(ctx["sources"], self.cfg.output_dir)  # improve tests
            )
            # .assign(
            #     _synced=lambda ctx: self.compiler.sync_tests_from_dir(self.cfg.output_dir)  # sync tests into Gradle
            # )
            # .assign(
            #     ok=lambda ctx: self.compiler.run_tests()  # run tests
            # )
            # .assign(
            #     coverage=lambda ctx: self.coverage.get_coverage()  # compute coverage
            # )
        )

        best_cov = 0.0
        ctx = {"sources": sources}
        for it in range(1, self.cfg.max_iterations + 1):
            self.logger.info(f"[LC] Iteration {it}/{self.cfg.max_iterations}")
            ctx = iteration_chain.invoke(ctx)
            cov = float(ctx.get("coverage", 0.0))
            best_cov = max(best_cov, cov)
            self.logger.info(f"Coverage after iteration {it}: {cov:.2f}% (best: {best_cov:.2f}%)")
            if cov >= self.cfg.coverage_threshold:
                self.logger.info("Coverage threshold met; stopping.")
                break
            time.sleep(1)

        return best_cov

    def _run_builtin(self) -> float:
        """Original built-in loop without LangChain."""
        # Step 1: read
        sources = self.reader.read()
        if not sources:
            self.logger.error("No Kotlin sources found to generate tests for.")
            return 0.0

        # Initial sync of sources into Gradle project
        self.compiler.sync_sources(sources)

        best_cov = 0.0
        for it in range(1, self.cfg.max_iterations + 1):
            self.logger.info(f"Iteration {it}/{self.cfg.max_iterations}")

            # Step 2: generate tests
            self.generator.generate_for_sources(sources)

            # Step 2.5: improve tests using static analysis
            improved = self.improver.improve_dir(sources, self.cfg.output_dir)
            if improved:
                self.logger.info(f"Static analysis improved {improved} test file(s)")

            # Step 3: sync tests and run compilation/tests
            self.compiler.sync_tests_from_dir(self.cfg.output_dir)
            ok = self.compiler.run_tests()
            if not ok:
                self.logger.warning("Compilation/tests failed; continuing to next iteration if any")

            # Step 4: coverage
            cov = self.coverage.get_coverage()
            best_cov = max(best_cov, cov)
            self.logger.info(f"Coverage after iteration {it}: {cov:.2f}% (best: {best_cov:.2f}%)")
            if cov >= self.cfg.coverage_threshold:
                self.logger.info("Coverage threshold met; stopping.")
                break

            # Optional backoff
            time.sleep(1)

        return best_cov

    def run(self) -> float:
        # Use LangChain-run orchestrator if enabled and available
        if self.cfg.use_langchain:
            return self._run_with_langchain()
        # Otherwise, use the original built-in loop
        return self._run_builtin()


def run_pipeline(
    source_dir: str = "input-src",
    output_dir: str = "output-test",
    gradle_project_dir: str = "validation-system/gradle-project",
    coverage_threshold: float = 80.0,
    max_iterations: int = 1,
    use_langchain: bool = True,
) -> float:
    orchestrator = MultiAgentOrchestrator(
        PipelineConfig(
            source_dir=source_dir,
            output_dir=output_dir,
            gradle_project_dir=gradle_project_dir,
            coverage_threshold=coverage_threshold,
            max_iterations=max_iterations,
            use_langchain=use_langchain,
        )
    )
    return orchestrator.run()
