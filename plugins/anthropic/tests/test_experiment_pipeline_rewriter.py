"""Unit tests for the PipelineRewriter ML experiment framework."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import flyte
import pytest

from flyteplugins.anthropic.experiment import PipelineRewriter
from flyteplugins.anthropic.experiment._types import ExperimentHistory, TrialRecord


# ---------------------------------------------------------------------------
# ExperimentHistory tests
# ---------------------------------------------------------------------------


def test_experiment_history_minimize_tracks_best():
    history = ExperimentHistory(goal="minimize")
    assert history.best_metric is None

    r1 = TrialRecord(iteration=0, task_name="t", metric=0.8, improved=False)
    history.add(r1)
    assert history.best_metric == 0.8
    assert r1.improved is True

    r2 = TrialRecord(iteration=1, task_name="t", metric=0.9, improved=False)
    history.add(r2)
    assert history.best_metric == 0.8  # 0.9 is worse for minimize
    assert r2.improved is False

    r3 = TrialRecord(iteration=2, task_name="t", metric=0.5, improved=False)
    history.add(r3)
    assert history.best_metric == 0.5  # 0.5 is better
    assert r3.improved is True


def test_experiment_history_maximize_tracks_best():
    history = ExperimentHistory(goal="maximize")

    r1 = TrialRecord(iteration=0, task_name="t", metric=0.3, improved=False)
    history.add(r1)
    assert history.best_metric == 0.3
    assert r1.improved is True

    r2 = TrialRecord(iteration=1, task_name="t", metric=0.7, improved=False)
    history.add(r2)
    assert history.best_metric == 0.7
    assert r2.improved is True

    r3 = TrialRecord(iteration=2, task_name="t", metric=0.5, improved=False)
    history.add(r3)
    assert history.best_metric == 0.7  # 0.5 is worse for maximize
    assert r3.improved is False


def test_experiment_history_skips_none_metrics():
    history = ExperimentHistory(goal="minimize")
    r = TrialRecord(iteration=0, task_name="t", metric=None, improved=False, error="fail")
    history.add(r)
    assert history.best_metric is None
    assert r.improved is False


def test_experiment_history_to_json():
    history = ExperimentHistory(goal="minimize")
    history.add(TrialRecord(iteration=0, task_name="train", metric=0.6, improved=False))
    data = json.loads(history.to_json())
    assert data["goal"] == "minimize"
    assert data["best_metric"] == 0.6
    assert data["total_trials"] == 1
    assert len(data["trials"]) == 1
    assert data["trials"][0]["task_name"] == "train"


# ---------------------------------------------------------------------------
# Helpers to build mock Flyte tasks
# ---------------------------------------------------------------------------


def _make_mock_task(name: str, input_names: list[str], output_name: str = "o0") -> MagicMock:
    """Build a minimal mock that looks like an AsyncFunctionTaskTemplate."""
    mock_func = MagicMock(__name__=name)

    interface = MagicMock()
    interface.inputs = {k: (float, None) for k in input_names}
    interface.outputs = {output_name: float}

    task = MagicMock()
    task.func = mock_func
    task.interface = interface
    task.aio = AsyncMock()
    return task


# ---------------------------------------------------------------------------
# PipelineRewriter validation tests
# ---------------------------------------------------------------------------


def test_pipeline_rewriter_raises_if_metric_task_not_last():
    env = flyte.TaskEnvironment("test-env")

    @env.task
    async def task_a(x: float) -> float:
        return x

    @env.task
    async def task_b(x: float) -> float:
        return x

    with pytest.raises(ValueError, match="metric_task must be the last"):
        rewriter = PipelineRewriter(
            pipeline_tasks=[task_a, task_b],
            mutable_tasks=[task_a],
            metric_task=task_a,  # wrong: should be task_b
            goal="minimize",
            goal_description="Test",
        )
        import asyncio
        asyncio.run(rewriter.run())


def test_pipeline_rewriter_raises_if_mutable_not_in_pipeline():
    env = flyte.TaskEnvironment("test-env-b")

    @env.task
    async def task_a(x: float) -> float:
        return x

    @env.task
    async def task_b(x: float) -> float:
        return x

    @env.task
    async def task_c(x: float) -> float:
        return x

    with pytest.raises(ValueError, match="not in pipeline_tasks"):
        rewriter = PipelineRewriter(
            pipeline_tasks=[task_a, task_b],
            mutable_tasks=[task_c],  # task_c is not in pipeline_tasks
            metric_task=task_b,
            goal="minimize",
            goal_description="Test",
        )
        import asyncio
        asyncio.run(rewriter.run())


# ---------------------------------------------------------------------------
# Tool-level unit tests (read_task_source, syntax validation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_task_source_returns_error_for_non_mutable():
    """read_task_source returns an error string for tasks not in mutable_tasks."""
    env = flyte.TaskEnvironment("test-env-c")

    @env.task
    async def load(x: float) -> float:
        return x * 2

    @env.task
    async def train(x: float) -> float:
        return x * 3

    @env.task
    async def evaluate(x: float) -> float:
        return x * 4

    rewriter = PipelineRewriter(
        pipeline_tasks=[load, train, evaluate],
        mutable_tasks=[train],
        metric_task=evaluate,
        goal="minimize",
        goal_description="Test",
    )

    # We need to reach into the run() coroutine's closure.  The cleanest way is
    # to run the whole thing with a mocked run_agent that calls the tool directly.
    captured_tools = {}

    async def fake_run_agent(prompt, tools, system, model, max_iterations, **kwargs):
        for t in tools:
            captured_tools[t.name] = t
        return "DONE"

    with patch("flyteplugins.anthropic.experiment._pipeline_rewriter.run_agent", side_effect=fake_run_agent):
        with patch.object(load, "aio", new_callable=AsyncMock, return_value=1.0):
            await rewriter.run()

    read_source_tool = captured_tools.get("read_task_source")
    assert read_source_tool is not None

    result = await read_source_tool.execute(task_name="load")
    assert "not in mutable_tasks" in result or "not mutable" in result.lower() or "Error" in result


@pytest.mark.asyncio
async def test_run_pipeline_trial_rejects_syntax_error():
    """run_pipeline_trial returns a JSON error for syntactically invalid new_source."""
    env = flyte.TaskEnvironment("test-env-d")

    @env.task
    async def load() -> float:
        return 1.0

    @env.task
    async def train(x: float) -> float:
        return x

    @env.task
    async def evaluate(x: float) -> float:
        return x

    rewriter = PipelineRewriter(
        pipeline_tasks=[load, train, evaluate],
        mutable_tasks=[train],
        metric_task=evaluate,
        goal="minimize",
        goal_description="Test",
    )

    captured_tools = {}

    async def fake_run_agent(prompt, tools, system, model, max_iterations, **kwargs):
        for t in tools:
            captured_tools[t.name] = t
        return "DONE"

    with patch("flyteplugins.anthropic.experiment._pipeline_rewriter.run_agent", side_effect=fake_run_agent):
        with patch.object(load, "aio", new_callable=AsyncMock, return_value=1.0):
            await rewriter.run()

    trial_tool = captured_tools.get("run_pipeline_trial")
    assert trial_tool is not None

    result_json = await trial_tool.execute(task_name="train", new_source="def (: bad syntax !!!")
    result = json.loads(result_json)
    assert result["success"] is False
    assert "SyntaxError" in result["error"]


@pytest.mark.asyncio
async def test_run_pipeline_trial_rejects_non_mutable_task():
    """run_pipeline_trial rejects a task_name that is not in mutable_tasks."""
    env = flyte.TaskEnvironment("test-env-e")

    @env.task
    async def load() -> float:
        return 1.0

    @env.task
    async def train(x: float) -> float:
        return x

    @env.task
    async def evaluate(x: float) -> float:
        return x

    rewriter = PipelineRewriter(
        pipeline_tasks=[load, train, evaluate],
        mutable_tasks=[train],
        metric_task=evaluate,
        goal="minimize",
        goal_description="Test",
    )

    captured_tools = {}

    async def fake_run_agent(prompt, tools, system, model, max_iterations, **kwargs):
        for t in tools:
            captured_tools[t.name] = t
        return "DONE"

    with patch("flyteplugins.anthropic.experiment._pipeline_rewriter.run_agent", side_effect=fake_run_agent):
        with patch.object(load, "aio", new_callable=AsyncMock, return_value=1.0):
            await rewriter.run()

    trial_tool = captured_tools.get("run_pipeline_trial")
    result_json = await trial_tool.execute(task_name="evaluate", new_source="o0 = x * 2")
    result = json.loads(result_json)
    assert result["success"] is False
    assert "not mutable" in result["error"]


# ---------------------------------------------------------------------------
# Full run() smoke test with a toy pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_rewriter_run_full_smoke():
    """Smoke test: run() with mocked sandbox and run_agent completes without error."""
    env = flyte.TaskEnvironment("test-env-smoke")

    @env.task
    async def load() -> float:
        return 2.0

    @env.task
    async def compute(x: float) -> float:
        return x * x

    @env.task
    async def metric(x: float) -> float:
        return x

    mock_sandbox = MagicMock()
    mock_sandbox.run = MagicMock()
    mock_sandbox.run.aio = AsyncMock(return_value=3.0)

    rewriter = PipelineRewriter(
        pipeline_tasks=[load, compute, metric],
        mutable_tasks=[compute],
        metric_task=metric,
        goal="minimize",
        goal_description="Minimize the output.",
        iterations=2,
        trial_packages=["numpy"],
    )

    trial_results = []

    async def fake_run_agent(prompt, tools, system, model, max_iterations, **kwargs):
        # Simulate the agent calling run_pipeline_trial once with valid code.
        tool_map = {t.name: t for t in tools}
        trial_tool = tool_map["run_pipeline_trial"]
        result = await trial_tool.execute(task_name="compute", new_source="o0 = x - 1.0")
        trial_results.append(json.loads(result))
        return "Experiment complete."

    with patch("flyteplugins.anthropic.experiment._pipeline_rewriter.flyte.sandbox.create", return_value=mock_sandbox):
        with patch("flyteplugins.anthropic.experiment._pipeline_rewriter.run_agent", side_effect=fake_run_agent):
            with patch.object(load, "aio", new_callable=AsyncMock, return_value=2.0):
                with patch.object(metric, "aio", new_callable=AsyncMock, return_value=3.0):
                    output = await rewriter.run()

    assert len(trial_results) == 1
    assert trial_results[0]["success"] is True
    assert trial_results[0]["metric"] == 3.0

    result_data = json.loads(output)
    assert result_data["total_trials"] == 1
    assert result_data["best_metric"] == 3.0
    assert "agent_summary" in result_data
