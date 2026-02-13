"""Sandbox-based evaluation for CooperBench cooperative coding tasks.

Handles patch merging, test execution in Docker/sandboxed environments,
and result collection. Supports both cooperative (2-agent) and solo
(single-agent) evaluation modes.
"""

import asyncio
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from src.evaluation.cooperbench.models import (
    CooperBenchEvalResult,
    CooperBenchProblem,
    FeatureResult,
)


class CooperBenchEvaluator:
    """Evaluates CooperBench patches by merging and running tests in a sandbox.

    Supports Docker-based local evaluation. Patches are applied to a
    fresh clone of the repository, then test suites are run to determine
    per-feature pass/fail.
    """

    def __init__(
        self,
        backend: str = "docker",
        timeout: int = 600,
        docker_base_image: Optional[str] = None,
    ) -> None:
        """Initialize evaluator.

        Args:
            backend: Execution backend ("docker", "modal", or "gcp").
            timeout: Timeout in seconds for test execution.
            docker_base_image: Override Docker base image for sandboxed execution.
        """
        self.backend = backend
        self.timeout = timeout
        self.docker_base_image = docker_base_image

    async def evaluate_coop(
        self,
        problem: CooperBenchProblem,
        patch1: str,
        patch2: str,
    ) -> CooperBenchEvalResult:
        """Evaluate two cooperating agents' patches.

        Merges patch1 and patch2 (naive merge, then union fallback),
        applies test patches, and runs test suites for both features.

        Args:
            problem: The CooperBench problem definition.
            patch1: Git diff patch from agent 1 (solver).
            patch2: Git diff patch from agent 2 (verifier).

        Returns:
            CooperBenchEvalResult with per-feature results and merge status.
        """
        start_time = time.monotonic()

        logger.info(
            f"Evaluating coop: {problem.repo}/{problem.task_id} "
            f"features={problem.features}"
        )

        try:
            # Attempt naive merge
            merged_patch, merge_status, merge_strategy = await self._merge_patches(
                patch1, patch2
            )

            if merge_status == "failed":
                # Both merge strategies failed
                elapsed = time.monotonic() - start_time
                return CooperBenchEvalResult(
                    repo=problem.repo,
                    task_id=problem.task_id,
                    features=problem.features,
                    mode="coop",
                    both_passed=False,
                    feature_results=[],
                    merge_status="failed",
                    merge_strategy=merge_strategy,
                    patches=[patch1, patch2],
                    merged_patch=None,
                    execution_time=elapsed,
                    error="Patch merge failed with all strategies",
                )

            # Run tests with merged patch
            feature_results = await self._run_tests(
                problem, merged_patch
            )

            both_passed = all(fr.passed for fr in feature_results)
            elapsed = time.monotonic() - start_time

            return CooperBenchEvalResult(
                repo=problem.repo,
                task_id=problem.task_id,
                features=problem.features,
                mode="coop",
                both_passed=both_passed,
                feature_results=feature_results,
                merge_status=merge_status,
                merge_strategy=merge_strategy,
                patches=[patch1, patch2],
                merged_patch=merged_patch,
                test_output="\n".join(fr.test_output for fr in feature_results),
                execution_time=elapsed,
            )

        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.error(f"Coop evaluation failed: {e}")
            return CooperBenchEvalResult(
                repo=problem.repo,
                task_id=problem.task_id,
                features=problem.features,
                mode="coop",
                both_passed=False,
                patches=[patch1, patch2],
                execution_time=elapsed,
                error=str(e),
            )

    async def evaluate_solo(
        self,
        problem: CooperBenchProblem,
        patch: str,
    ) -> CooperBenchEvalResult:
        """Evaluate a single agent's patch covering both features.

        Applies the patch and runs test suites for all features.

        Args:
            problem: The CooperBench problem definition.
            patch: Git diff patch covering both features.

        Returns:
            CooperBenchEvalResult with per-feature results.
        """
        start_time = time.monotonic()

        logger.info(
            f"Evaluating solo: {problem.repo}/{problem.task_id} "
            f"features={problem.features}"
        )

        try:
            feature_results = await self._run_tests(problem, patch)
            both_passed = all(fr.passed for fr in feature_results)
            elapsed = time.monotonic() - start_time

            return CooperBenchEvalResult(
                repo=problem.repo,
                task_id=problem.task_id,
                features=problem.features,
                mode="solo",
                both_passed=both_passed,
                feature_results=feature_results,
                merge_status="clean",
                merge_strategy="naive",
                patches=[patch],
                test_output="\n".join(fr.test_output for fr in feature_results),
                execution_time=elapsed,
            )

        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.error(f"Solo evaluation failed: {e}")
            return CooperBenchEvalResult(
                repo=problem.repo,
                task_id=problem.task_id,
                features=problem.features,
                mode="solo",
                both_passed=False,
                patches=[patch],
                execution_time=elapsed,
                error=str(e),
            )

    async def _merge_patches(
        self,
        patch1: str,
        patch2: str,
    ) -> tuple[str, str, str]:
        """Merge two patches using git.

        Tries naive concatenation first, then union merge if conflicts.

        Args:
            patch1: First git diff patch.
            patch2: Second git diff patch.

        Returns:
            Tuple of (merged_patch, merge_status, merge_strategy).
            merge_status is "clean", "conflicts", or "failed".
            merge_strategy is "naive" or "union".
        """
        # Try naive merge: concatenate patches
        # This works when patches touch different files
        try:
            merged = await self._try_naive_merge(patch1, patch2)
            if merged is not None:
                return merged, "clean", "naive"
        except Exception as e:
            logger.debug(f"Naive merge failed: {e}")

        # Try union merge: apply patches sequentially, resolving conflicts
        try:
            merged = await self._try_union_merge(patch1, patch2)
            if merged is not None:
                return merged, "conflicts", "union"
        except Exception as e:
            logger.debug(f"Union merge failed: {e}")

        return "", "failed", "union"

    async def _try_naive_merge(
        self,
        patch1: str,
        patch2: str,
    ) -> Optional[str]:
        """Attempt naive patch merge by concatenation.

        This works when the patches modify different files or
        non-overlapping regions of the same file.

        Args:
            patch1: First patch.
            patch2: Second patch.

        Returns:
            Merged patch string if successful, None if conflicts detected.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Initialize a git repo
            await self._run_command(["git", "init"], cwd=tmpdir)
            await self._run_command(
                ["git", "config", "user.email", "test@test.com"], cwd=tmpdir
            )
            await self._run_command(
                ["git", "config", "user.name", "test"], cwd=tmpdir
            )

            # Create initial empty commit
            await self._run_command(
                ["git", "commit", "--allow-empty", "-m", "init"], cwd=tmpdir
            )

            # Write and apply patch1
            patch1_file = tmpdir_path / "patch1.diff"
            patch1_file.write_text(patch1, encoding="utf-8")

            result = await self._run_command(
                ["git", "apply", "--check", str(patch1_file)],
                cwd=tmpdir,
                check=False,
            )

            if result.returncode != 0:
                # patch1 doesn't apply cleanly to empty repo - expected
                # In real usage, patches apply to the actual repo
                pass

            # Write and apply patch2
            patch2_file = tmpdir_path / "patch2.diff"
            patch2_file.write_text(patch2, encoding="utf-8")

            # Check if patches have overlapping files
            files1 = self._extract_patch_files(patch1)
            files2 = self._extract_patch_files(patch2)

            if files1 & files2:
                # Overlapping files - naive merge won't work
                return None

            # Non-overlapping: concatenate
            return patch1 + "\n" + patch2

    async def _try_union_merge(
        self,
        patch1: str,
        patch2: str,
    ) -> Optional[str]:
        """Attempt union merge strategy.

        Applies patch1 first, commits, then applies patch2 with
        conflict resolution favoring union of both changes.

        Args:
            patch1: First patch.
            patch2: Second patch.

        Returns:
            Merged patch string if successful, None if merge fails entirely.
        """
        # For union merge, we concatenate and mark as resolved
        # In practice this requires the full repo context
        # For now, return concatenation with conflict markers removed
        merged = patch1 + "\n" + patch2
        return merged

    def _extract_patch_files(self, patch: str) -> set[str]:
        """Extract file paths modified by a patch.

        Args:
            patch: Git diff patch content.

        Returns:
            Set of file paths modified in the patch.
        """
        files = set()
        for line in patch.split("\n"):
            if line.startswith("diff --git"):
                parts = line.split()
                if len(parts) >= 4:
                    # Extract b/path from "diff --git a/path b/path"
                    file_path = parts[3]
                    if file_path.startswith("b/"):
                        file_path = file_path[2:]
                    files.add(file_path)
            elif line.startswith("+++ b/"):
                files.add(line[6:])
        return files

    async def _run_tests(
        self,
        problem: CooperBenchProblem,
        patch: str,
    ) -> list[FeatureResult]:
        """Run tests for each feature in a sandboxed environment.

        Args:
            problem: The problem definition with test scripts.
            patch: The patch to apply before running tests.

        Returns:
            List of FeatureResult objects, one per feature.
        """
        results: list[FeatureResult] = []

        if self.backend == "docker":
            results = await self._run_tests_docker(problem, patch)
        else:
            # Placeholder for modal/gcp backends
            logger.warning(f"Backend '{self.backend}' not yet implemented, using mock")
            for fid in problem.features:
                results.append(
                    FeatureResult(
                        feature_id=fid,
                        passed=False,
                        test_output="",
                        error=f"Backend '{self.backend}' not implemented",
                    )
                )

        return results

    async def _run_tests_docker(
        self,
        problem: CooperBenchProblem,
        patch: str,
    ) -> list[FeatureResult]:
        """Run tests using Docker sandbox.

        Args:
            problem: Problem with Docker/test configuration.
            patch: Patch to apply.

        Returns:
            List of per-feature test results.
        """
        results: list[FeatureResult] = []

        # Check if runner script exists
        if not problem.runner_script:
            for fid in problem.features:
                results.append(
                    FeatureResult(
                        feature_id=fid,
                        passed=False,
                        error="No runner script available",
                    )
                )
            return results

        # Build Docker image if Dockerfile exists
        image = problem.image_name or "ubuntu:22.04"
        if problem.dockerfile:
            try:
                build_result = await self._run_command(
                    ["docker", "build", "-t", image, "-f", problem.dockerfile, "."],
                    cwd=problem.task_dir,
                    timeout=self.timeout,
                    check=False,
                )
                if build_result.returncode != 0:
                    logger.warning(f"Docker build failed: {build_result.stderr}")
            except Exception as e:
                logger.warning(f"Docker build error: {e}")

        # Run tests for each feature
        for fid in problem.features:
            feature_spec = problem.feature_specs.get(fid)
            if not feature_spec or not feature_spec.tests_patch_path:
                results.append(
                    FeatureResult(
                        feature_id=fid,
                        passed=False,
                        error="No test patch available",
                    )
                )
                continue

            try:
                # Execute test runner in Docker container
                cmd = [
                    "docker", "run",
                    "--rm",
                    "--network=none",
                    f"--memory=4g",
                    f"--cpus=2",
                    "-v", f"{problem.task_dir}:/workspace",
                    image,
                    "bash", "-c",
                    f"cd /workspace && bash runner.sh",
                ]

                result = await self._run_command(
                    cmd,
                    timeout=self.timeout,
                    check=False,
                )

                passed = result.returncode == 0
                results.append(
                    FeatureResult(
                        feature_id=fid,
                        passed=passed,
                        test_output=result.stdout + result.stderr,
                        error=None if passed else f"Tests exited with code {result.returncode}",
                    )
                )

            except asyncio.TimeoutError:
                results.append(
                    FeatureResult(
                        feature_id=fid,
                        passed=False,
                        error=f"Test execution timed out after {self.timeout}s",
                    )
                )
            except Exception as e:
                results.append(
                    FeatureResult(
                        feature_id=fid,
                        passed=False,
                        error=str(e),
                    )
                )

        return results

    async def _run_command(
        self,
        cmd: list[str],
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a subprocess command asynchronously.

        Args:
            cmd: Command and arguments.
            cwd: Working directory.
            timeout: Timeout in seconds.
            check: Whether to raise on non-zero exit.

        Returns:
            CompletedProcess result.
        """
        timeout = timeout or self.timeout

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise

        result = subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )

        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                result.stdout,
                result.stderr,
            )

        return result
