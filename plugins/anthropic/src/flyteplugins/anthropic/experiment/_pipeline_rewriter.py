"""Pipeline Rewriter — Agent optimizes task function bodies in a Flyte pipeline.

A Claude agent is equipped with three tools:
  - read_task_source   : read the current source of any mutable task
  - run_pipeline_trial : swap in new code for one task and measure the metric
  - read_experiment_log: inspect the full history of all trials

On each iteration the agent proposes an incremental change, runs the modified
pipeline end-to-end through a flyte.sandbox, and keeps or discards the change
based on the measured metric — the same loop as karpathy/autoresearch but
operating at the Flyte task level rather than a monolithic script.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import flyte
import flyte.sandbox

from flyteplugins.anthropic.agents import function_tool, run_agent

from ._types import ExperimentHistory, Goal, TrialRecord

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an ML experiment optimization agent integrated with a Flyte pipeline.

Goal: {goal_description}
Metric direction: {goal} — {direction} values are better.
Mutable tasks you may rewrite: {mutable_task_names}

You have three tools:

read_task_source(task_name)
  Returns the current Python source of a mutable task function.

run_pipeline_trial(task_name, new_source)
  Replaces the implementation of task_name with new_source and runs the full
  pipeline. new_source must be a Python SCRIPT (not a function definition).
  Flyte will pre-inject these local variables for you: {input_variables_hint}
  Your script must assign the result to: {output_variable_hint}
  Returns JSON with keys: success, metric, improved, best_metric (or error).

read_experiment_log()
  Returns the full JSON history of all trials run so far.

Strategy:
- Read the task source first to understand the current implementation.
- Propose one focused change at a time (optimizer, architecture, data augmentation,
  regularisation, etc.).
- On syntax or runtime errors, read the error message and fix the code.
- When a change does not improve the metric, try a different direction.
- Reply with the word DONE when you have exhausted productive ideas.
"""


@dataclass
class PipelineRewriter:
    """Optimizes Flyte task implementations via a Claude agent loop.

    The user declares an ordered pipeline (list of tasks that form a linear chain)
    and a whitelist of *mutable* tasks the agent may rewrite. On each iteration the
    agent reads the source of a task, proposes a modified script, and the framework
    runs the full pipeline with the modification in an isolated sandbox, measuring the
    scalar metric returned by the final task.

    Example::

        experiment = PipelineRewriter(
            pipeline_tasks=[load_data, train_model, evaluate],
            mutable_tasks=[train_model],
            metric_task=evaluate,
            goal="minimize",
            goal_description="Minimize validation bits-per-byte.",
            iterations=20,
            trial_packages=["torch", "transformers"],
        )

        @agent_env.task
        async def run_experiment() -> str:
            return await experiment.run()

        run = flyte.run(run_experiment)

    Pipeline contract:
    - ``pipeline_tasks`` must be ordered: each task's *first* typed parameter
      receives the return value of the preceding task.
    - ``metric_task`` must be the final entry in ``pipeline_tasks`` and must
      return a scalar ``float``.
    - ``mutable_tasks`` must be a subset of ``pipeline_tasks``.
    - Tasks before the first mutable task are executed *once* and their outputs
      are cached for reuse across trials.
    """

    pipeline_tasks: list
    """Ordered list of AsyncFunctionTaskTemplate forming a linear pipeline chain."""

    mutable_tasks: list
    """Subset of pipeline_tasks whose implementations the agent may rewrite."""

    metric_task: Any
    """The final task in the pipeline; must return a scalar float metric."""

    goal: Goal
    """'minimize' or 'maximize'."""

    goal_description: str
    """Natural language description of the experiment objective for the agent."""

    iterations: int = 20
    """Maximum number of pipeline trials to run."""

    model: str = "claude-opus-4-6"
    """Claude model to use for the agent."""

    trial_packages: list[str] = field(default_factory=list)
    """Pip packages to install in the sandbox container for each trial."""

    async def run(self) -> str:
        """Execute the optimization loop.

        Must be called from within a ``@env.task``-decorated async function so
        that downstream task and sandbox calls are properly orchestrated by Flyte.

        Returns a JSON summary string containing the best metric and trial history.
        """
        mutable_names: set[str] = {t.func.__name__ for t in self.mutable_tasks}
        mutable_map: dict[str, Any] = {t.func.__name__: t for t in self.mutable_tasks}
        pipeline_names: list[str] = [t.func.__name__ for t in self.pipeline_tasks]

        # Validate
        last_name = pipeline_names[-1] if pipeline_names else ""
        if self.metric_task.func.__name__ != last_name:
            raise ValueError(
                f"metric_task must be the last entry in pipeline_tasks. "
                f"Got '{self.metric_task.func.__name__}', expected '{last_name}'."
            )
        unknown = mutable_names - set(pipeline_names)
        if unknown:
            raise ValueError(f"mutable_tasks contains tasks not in pipeline_tasks: {unknown}")

        # Find index of the first mutable task; pre-run everything before it once.
        mutable_indices = [i for i, t in enumerate(self.pipeline_tasks) if t.func.__name__ in mutable_names]
        min_mutable_idx = min(mutable_indices)
        upstream_output = await self._run_chain(self.pipeline_tasks[:min_mutable_idx])

        history = ExperimentHistory(goal=self.goal)
        trial_counter = [0]  # list so the closure can mutate it

        # Build system-prompt hints from the first mutable task's interface.
        first_mutable = self.mutable_tasks[0]
        input_vars = list(first_mutable.interface.inputs.keys())
        output_vars = list(first_mutable.interface.outputs.keys())
        input_hint = ", ".join(f"`{v}`" for v in input_vars) if input_vars else "none (no inputs)"
        output_hint = ", ".join(f"`{v}`" for v in output_vars)

        # --- Tool definitions (closures over shared mutable state) --------------

        def read_task_source(task_name: str) -> str:
            """Read the current source code of a mutable pipeline task.

            Args:
                task_name: Name of the mutable task to read.

            Returns:
                The Python source code of the task function.
            """
            if task_name not in mutable_map:
                return f"Error: '{task_name}' is not in mutable_tasks. Allowed: {sorted(mutable_map)}"
            return inspect.getsource(mutable_map[task_name].func)

        async def run_pipeline_trial(task_name: str, new_source: str) -> str:
            """Run the full pipeline with a modified task implementation.

            Replaces task_name with new_source, executes the pipeline end-to-end
            in an isolated Docker sandbox, and returns the scalar metric.

            Args:
                task_name: Name of the mutable task to replace.
                new_source: Python script body for the replacement (see system prompt
                    for the exact variable injection / output assignment contract).

            Returns:
                JSON string with keys: success, metric, improved, best_metric, error.
            """
            if task_name not in mutable_names:
                return json.dumps({
                    "success": False,
                    "error": f"'{task_name}' is not mutable. Allowed: {sorted(mutable_names)}",
                })

            try:
                ast.parse(new_source)
            except SyntaxError as exc:
                return json.dumps({"success": False, "error": f"SyntaxError in new_source: {exc}"})

            task = mutable_map[task_name]
            task_idx = pipeline_names.index(task_name)
            inputs_types = {k: v[0] for k, v in task.interface.inputs.items()}
            outputs_types = task.interface.outputs
            trial_n = trial_counter[0]
            trial_counter[0] += 1

            try:
                sandbox = flyte.sandbox.create(
                    name=f"exp-{task_name}-t{trial_n}",
                    code=new_source,
                    inputs=inputs_types,
                    outputs=outputs_types,
                    packages=self.trial_packages,
                )

                # Run any tasks between min_mutable_idx and task_idx using originals.
                inter_output = upstream_output
                for t in self.pipeline_tasks[min_mutable_idx:task_idx]:
                    inter_output = await self._call_next(t, inter_output)

                # Run the sandboxed modified task.
                if inputs_types:
                    first_input = next(iter(inputs_types))
                    sandbox_result = await sandbox.run.aio(**{first_input: inter_output})
                else:
                    sandbox_result = await sandbox.run.aio()

                # Run downstream tasks after the modified task.
                downstream_result = await self._run_chain_from(
                    self.pipeline_tasks[task_idx + 1 :], sandbox_result
                )
                metric = float(downstream_result)

                record = TrialRecord(
                    iteration=trial_n,
                    task_name=task_name,
                    metric=metric,
                    improved=False,
                )
                history.add(record)

                return json.dumps({
                    "success": True,
                    "metric": metric,
                    "improved": record.improved,
                    "best_metric": history.best_metric,
                })

            except Exception as exc:
                logger.exception("Trial %d ('%s') failed", trial_n, task_name)
                record = TrialRecord(
                    iteration=trial_n,
                    task_name=task_name,
                    metric=None,
                    improved=False,
                    error=str(exc),
                )
                history.add(record)
                return json.dumps({"success": False, "error": str(exc)})

        def read_experiment_log() -> str:
            """Return the full history of all experiment trials as JSON.

            Returns:
                JSON string with goal, best_metric, total_trials, and per-trial records.
            """
            return history.to_json()

        # --- Agent loop --------------------------------------------------------

        system_prompt = _SYSTEM_PROMPT.format(
            goal_description=self.goal_description,
            goal=self.goal,
            direction="lower" if self.goal == "minimize" else "higher",
            mutable_task_names=", ".join(sorted(mutable_names)),
            input_variables_hint=input_hint,
            output_variable_hint=output_hint,
        )

        agent_response = await run_agent(
            prompt=(
                f"Optimize the pipeline.\n"
                f"Goal: {self.goal_description}\n"
                f"Current best metric: {history.best_metric} (no trials yet)"
            ),
            tools=[
                function_tool(read_task_source),
                function_tool(run_pipeline_trial),
                function_tool(read_experiment_log),
            ],
            system=system_prompt,
            model=self.model,
            max_iterations=self.iterations * 4,
        )

        return json.dumps({
            "best_metric": history.best_metric,
            "total_trials": len(history.records),
            "agent_summary": agent_response,
            "history": json.loads(history.to_json()),
        }, indent=2)

    # --- Private helpers -------------------------------------------------------

    async def _run_chain(self, tasks: list) -> Any:
        """Run a linear chain of tasks, passing each output as the next task's input."""
        result = None
        for task in tasks:
            result = await self._call_next(task, result)
        return result

    async def _run_chain_from(self, tasks: list, initial: Any) -> Any:
        """Run a chain starting with *initial* as the input for the first task."""
        result = initial
        for task in tasks:
            result = await self._call_next(task, result)
        return result

    async def _call_next(self, task: Any, prev_result: Any) -> Any:
        """Call a single task, forwarding prev_result as its first typed input."""
        inputs = task.interface.inputs
        if not inputs or prev_result is None:
            return await task.aio()
        first_input_name = next(iter(inputs))
        return await task.aio(**{first_input_name: prev_result})
