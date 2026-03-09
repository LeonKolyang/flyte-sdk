"""
Pipeline Rewriter — Autonomous ML Experiment Optimization
==========================================================

Demonstrates how to use ``PipelineRewriter`` to autonomously optimize a Flyte
pipeline.  A Claude agent is given tools to:

  1. Read the current source of any ``mutable_tasks`` function.
  2. Propose a rewritten implementation as a Python script.
  3. Run the full pipeline end-to-end with the rewrite in an isolated sandbox.
  4. Inspect the accumulated experiment history.

The agent iterates until the metric stops improving or the iteration budget is
exhausted — the same loop as karpathy/autoresearch, applied at the Flyte task
level.

Usage
-----
    flyte run pipeline_rewriter_example.py run_experiment

Pipeline
--------
  generate_data  →  compute  →  measure

  generate_data : no inputs, returns a float seed value.
  compute       : takes the seed, runs a trivial calculation, returns float.
                  ← this is the MUTABLE task the agent will optimise.
  measure       : returns the final metric (the value to minimise).

Goal: minimise the value returned by ``measure``.  The optimal strategy is
to subtract as large a constant as possible from the seed — the agent should
discover this pattern within a few iterations.

Running Locally
---------------
Set the ANTHROPIC_API_KEY environment variable and run:

    ANTHROPIC_API_KEY=sk-... flyte run pipeline_rewriter_example.py run_experiment

On a Union cluster the key should be stored as a secret and referenced via:

    flyte.Secret(key="anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")
"""

from pathlib import Path

import flyte
import flyte.sandbox

from flyteplugins.anthropic.experiment import PipelineRewriter

# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

# Lightweight CPU environment for the orchestrating agent task.
# Needs: anthropic SDK, pydantic-monty (for flyte.sandbox).
agent_env = flyte.TaskEnvironment(
    name="experiment-agent",
    image=(
        flyte.Image.from_debian_base(name="experiment-agent-image").with_pip_packages(
            "anthropic",
        )
    ),
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    secrets=[
        # Store your Anthropic API key as a Union secret named "anthropic-api-key"
        # and it will be injected as the ANTHROPIC_API_KEY environment variable.
        flyte.Secret(key="anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
    ],
)

# The trial environment is used by the sandbox that runs agent-proposed code.
# For a real workload add GPU resources and ML packages here:
#
#   resources=flyte.Resources(gpu="A10G:1", cpu=8, memory="32Gi")
#   trial_packages=["torch", "transformers", "datasets"]
TRIAL_PACKAGES: list[str] = []

# ---------------------------------------------------------------------------
# Pipeline tasks
# ---------------------------------------------------------------------------

pipeline_env = flyte.TaskEnvironment(
    name="experiment-pipeline",
    image=flyte.Image.from_debian_base(name="experiment-pipeline-image"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)


@pipeline_env.task
async def generate_data() -> float:
    """Generate a fixed seed value for the pipeline.

    Returns:
        A constant seed used by the downstream compute task.
    """
    return 10.0


@pipeline_env.task
async def compute(x: float) -> float:
    """Transform the input seed value.

    This is the task the agent will optimise.  The goal is to minimise the
    metric returned by ``measure``, which is equal to the return value of this
    task.  The agent should discover that subtracting a large constant is the
    optimal strategy.

    The sandbox will pre-inject the variable ``x`` (the input) and expects
    the result to be assigned to ``o0`` (the single output).

    Args:
        x: Input seed from generate_data.

    Returns:
        A transformed float value.
    """
    # Baseline: identity transform (no improvement).
    o0 = x
    return o0


@pipeline_env.task
async def measure(x: float) -> float:
    """Return the scalar metric to be minimised.

    In a real pipeline this would be a validation loss, perplexity, or any
    other quality metric.  Here it is just the identity to keep the example
    self-contained.

    Args:
        x: Output from the compute task.

    Returns:
        The scalar metric value.
    """
    return x


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

experiment = PipelineRewriter(
    pipeline_tasks=[generate_data, compute, measure],
    mutable_tasks=[compute],       # agent may only rewrite this task
    metric_task=measure,           # returns the scalar to optimise
    goal="minimize",
    goal_description=(
        "Minimise the value returned by the measure task.  "
        "Focus on the compute function: find a mathematical transformation "
        "of the input `x` that produces a smaller output."
    ),
    iterations=10,                 # max number of sandbox trials
    trial_packages=TRIAL_PACKAGES,
    model="claude-opus-4-6",
)


# ---------------------------------------------------------------------------
# Orchestrating task
# ---------------------------------------------------------------------------


@agent_env.task(report=True)
async def run_experiment() -> str:
    """Run the autonomous ML experiment optimisation loop.

    Calls PipelineRewriter.run() which drives a Claude agent to iteratively
    rewrite the ``compute`` task, measures the metric via the full pipeline,
    and returns a JSON summary of all trials.

    Returns:
        JSON string with best_metric, total_trials, agent_summary, and history.
    """
    return await experiment.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=Path(__file__).parent,
    )

    print("Launching ML experiment optimisation...")
    run = flyte.run(run_experiment)
    print(f"Run URL: {run.url}")
