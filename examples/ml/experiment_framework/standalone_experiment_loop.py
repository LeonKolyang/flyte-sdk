"""
Standalone Experiment Loop — Build Your Own Autoresearch-Style Optimizer
========================================================================

This file shows how to implement the same agent-driven ML experiment loop as
``PipelineRewriter`` WITHOUT the plugin abstraction — directly on top of the
three flyte-sdk v2 primitives a user has available:

  flyte.sandbox.create()                  run code in an isolated container
  flyteplugins.anthropic.run_agent()      drive a Claude agent loop
  flyteplugins.anthropic.function_tool()  expose any callable as a Claude tool

Compare with pipeline_rewriter_example.py which imports PipelineRewriter from
the plugin. This file contains the same logic written out flat — useful when
you want full control and customisation without inheriting from the framework.

When to use this pattern instead of PipelineRewriter
-----------------------------------------------------
- You want to customise the agent system prompt substantially.
- Your pipeline is not a simple linear chain (fan-out, conditional branches).
- You want custom early-stopping, ensemble logic, or trial parallelism.
- You simply prefer reading one flat file rather than a class hierarchy.

Pipeline
--------
  load_data  →  train_and_evaluate

  load_data          : generates a synthetic sklearn classification dataset
                       and returns it as a JSON-encoded string.
  train_and_evaluate : trains a classifier on the dataset and returns
                       cross-validated accuracy (to be MAXIMISED).
                       ← this is the function the agent will iteratively
                         rewrite and improve.

The agent starts with a trivial baseline (DummyClassifier), reads its own
source, proposes a better sklearn classifier, runs it in a sandbox, and keeps
the change if accuracy improves.

Usage
-----
    flyte run standalone_experiment_loop.py run_my_experiment

    # or directly:
    ANTHROPIC_API_KEY=sk-... python standalone_experiment_loop.py

Requirements (add to your pyproject.toml / requirements.txt)
-------------------------------------------------------------
    flyte
    flyteplugins-anthropic   # provides run_agent / function_tool
    scikit-learn             # needed inside trial sandboxes
    numpy                    # needed inside trial sandboxes
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import flyte
import flyte.sandbox
from flyteplugins.anthropic.agents import function_tool, run_agent

# ---------------------------------------------------------------------------
# 1. Define environments
#    (exactly as you would in any Flyte project)
# ---------------------------------------------------------------------------

# The agent orchestrator: lightweight CPU task, only needs anthropic SDK.
agent_env = flyte.TaskEnvironment(
    name="my-experiment-agent",
    image=(
        flyte.Image.from_debian_base(name="my-experiment-agent-image")
        .with_pip_packages("anthropic")
    ),
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    secrets=[
        # Store your key as a Union secret; it arrives as an env var.
        flyte.Secret(key="anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
    ],
)

# The pipeline tasks: use the same image as the agent here for simplicity;
# in a real project you would use a GPU image with your ML libraries.
pipeline_env = flyte.TaskEnvironment(
    name="my-experiment-pipeline",
    image=(
        flyte.Image.from_debian_base(name="my-experiment-pipeline-image")
        .with_pip_packages("scikit-learn", "numpy")
    ),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)

# Packages available to the agent-generated trial code (same as pipeline_env).
TRIAL_PACKAGES = ["scikit-learn", "numpy"]

# ---------------------------------------------------------------------------
# 2. Define your pipeline tasks
#    (normal @env.task functions — no changes needed to use the agent loop)
# ---------------------------------------------------------------------------


@pipeline_env.task
async def load_data() -> str:
    """Generate a synthetic binary classification dataset.

    Returns:
        JSON string with keys 'X' (list-of-lists) and 'y' (list of ints).
        This format is lightweight and easy for the sandbox to consume.
    """
    from sklearn.datasets import make_classification
    import numpy as np

    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )
    return json.dumps({"X": X.tolist(), "y": y.tolist()})


@pipeline_env.task
async def train_and_evaluate(dataset: str) -> float:
    """Baseline classifier: DummyClassifier (stratified random predictions).

    The agent will iteratively replace this implementation.  The sandbox
    pre-injects the variable ``dataset`` (str) and expects the result to be
    assigned to ``o0`` (float, the cross-validated accuracy).

    Args:
        dataset: JSON-encoded dict with keys 'X' and 'y'.

    Returns:
        Cross-validated accuracy (higher is better).
    """
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import cross_val_score

    data = json.loads(dataset)
    X = np.array(data["X"])
    y = np.array(data["y"])

    model = DummyClassifier(strategy="stratified", random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    o0 = float(scores.mean())
    return o0


# ---------------------------------------------------------------------------
# 3. The experiment loop — written flat, no abstraction class
#    This is the part that pipeline_rewriter_example.py wraps in a class.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an ML experiment optimisation agent.

Goal: maximise the cross-validated accuracy of train_and_evaluate.
Higher accuracy is better.

You have three tools:

read_current_source()
  Returns the current Python source of train_and_evaluate.

run_trial(new_source)
  Runs the full pipeline with new_source replacing train_and_evaluate.
  new_source is a Python SCRIPT (not a function definition). Flyte will
  pre-inject the variable `dataset` (JSON str with keys 'X' and 'y').
  Your script must assign the result accuracy to the variable `o0` (float).
  Returns JSON: {success, accuracy, improved, best_accuracy} or {success, error}.

read_history()
  Returns JSON history of all trials (accuracy, improved, error per trial).

Instructions:
- Start by reading the current source to understand the baseline.
- Propose improvements: try a different sklearn model, tune hyperparameters,
  add feature engineering (scaling, PCA, polynomial features), or use an
  ensemble. One focused change per trial.
- If a trial fails with an error, read the traceback and fix the code.
- If a trial does not improve accuracy, try a completely different approach.
- Reply DONE when you have exhausted productive ideas.
"""


@agent_env.task(report=True)
async def run_my_experiment(
    iterations: int = 15,
    model: str = "claude-opus-4-6",
) -> str:
    """Autonomous ML experiment optimisation loop.

    Drives a Claude agent to iteratively rewrite ``train_and_evaluate``,
    runs each candidate in an isolated sandbox, and accumulates results.

    Args:
        iterations: Maximum number of sandbox trials.
        model: Claude model to use for the agent.

    Returns:
        JSON summary: best_accuracy, total_trials, history, agent_summary.
    """
    # ------------------------------------------------------------------
    # 3a. Pre-run the frozen upstream task once.
    #     In a real pipeline you might have several upstream tasks;
    #     cache them all here before entering the agent loop.
    # ------------------------------------------------------------------
    dataset: str = await load_data.aio()

    # ------------------------------------------------------------------
    # 3b. Shared mutable state across tool calls.
    #     Using a list wrapping a single element lets inner closures
    #     mutate the binding without `nonlocal` (cleaner under Python 3.10).
    # ------------------------------------------------------------------
    best_accuracy: list[float | None] = [None]
    history: list[dict] = []
    trial_n: list[int] = [0]

    # ------------------------------------------------------------------
    # 3c. Tool definitions.
    #     These are plain Python callables — function_tool() inspects
    #     their signatures and docstrings to build the Claude tool schema.
    # ------------------------------------------------------------------

    def read_current_source() -> str:
        """Return the current source code of the train_and_evaluate function.

        Returns:
            Python source code string.
        """
        # inspect.getsource works on the underlying function, not the task wrapper.
        return inspect.getsource(train_and_evaluate.func)

    async def run_trial(new_source: str) -> str:
        """Replace train_and_evaluate with new_source and measure accuracy.

        Args:
            new_source: Python script. Inputs: `dataset` (str). Output: `o0` (float).

        Returns:
            JSON with keys: success, accuracy, improved, best_accuracy OR error.
        """
        # Syntax check before spinning up a Docker image — fast fail.
        try:
            ast.parse(new_source)
        except SyntaxError as exc:
            return json.dumps({"success": False, "error": f"SyntaxError: {exc}"})

        n = trial_n[0]
        trial_n[0] += 1

        # Create the sandbox: same input/output types as train_and_evaluate.
        sandbox = flyte.sandbox.create(
            name=f"trial-{n}",
            code=new_source,
            inputs={"dataset": str},
            outputs={"o0": float},
            packages=TRIAL_PACKAGES,
        )

        try:
            accuracy: float = await sandbox.run.aio(dataset=dataset)
            accuracy = float(accuracy)

            prev_best = best_accuracy[0]
            improved = prev_best is None or accuracy > prev_best
            if improved:
                best_accuracy[0] = accuracy

            record = {
                "trial": n,
                "accuracy": accuracy,
                "improved": improved,
                "best_accuracy": best_accuracy[0],
            }
            history.append(record)
            return json.dumps({"success": True, **record})

        except Exception as exc:
            record = {"trial": n, "accuracy": None, "improved": False, "error": str(exc)}
            history.append(record)
            return json.dumps({"success": False, "error": str(exc)})

    def read_history() -> str:
        """Return the full experiment history as JSON.

        Returns:
            JSON with best_accuracy, total_trials, and per-trial records.
        """
        return json.dumps(
            {
                "best_accuracy": best_accuracy[0],
                "total_trials": len(history),
                "trials": history,
            },
            indent=2,
        )

    # ------------------------------------------------------------------
    # 3d. Run the agent loop.
    #     run_agent drives the tool-calling conversation until the agent
    #     replies without calling any tools (stop_reason == "end_turn")
    #     or max_iterations is reached.
    # ------------------------------------------------------------------
    agent_summary = await run_agent(
        prompt=(
            f"Optimise train_and_evaluate to maximise cross-validated accuracy.\n"
            f"Current best accuracy: {best_accuracy[0]} (no trials yet — start fresh).\n"
            f"Budget: {iterations} trials."
        ),
        tools=[
            function_tool(read_current_source),
            function_tool(run_trial),
            function_tool(read_history),
        ],
        system=_SYSTEM_PROMPT,
        model=model,
        max_iterations=iterations * 4,  # tool-use turns, not trials
    )

    # ------------------------------------------------------------------
    # 3e. Return a structured summary for downstream tasks or the UI.
    # ------------------------------------------------------------------
    return json.dumps(
        {
            "best_accuracy": best_accuracy[0],
            "total_trials": len(history),
            "agent_summary": agent_summary,
            "history": history,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)
    print("Launching standalone experiment loop…")
    run = flyte.run(run_my_experiment, iterations=15)
    print(f"Run URL: {run.url}")
