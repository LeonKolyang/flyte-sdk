"""ML experiment framework for flyte-sdk v2.

Provides :class:`PipelineRewriter`, a high-level abstraction that lets ML
engineers define an experiment goal and then autonomously iterate on their
existing Flyte pipeline — the same loop as karpathy/autoresearch but operating
at the individual Flyte task level.

Example::

    from flyteplugins.anthropic.experiment import PipelineRewriter

    experiment = PipelineRewriter(
        pipeline_tasks=[load_data, train_model, evaluate],
        mutable_tasks=[train_model],
        metric_task=evaluate,
        goal="minimize",
        goal_description="Minimize validation bits-per-byte on the LM.",
        iterations=20,
        trial_packages=["torch", "transformers"],
    )

    @agent_env.task
    async def run_experiment() -> str:
        return await experiment.run()

    run = flyte.run(run_experiment)
"""

from ._pipeline_rewriter import PipelineRewriter
from ._types import ExperimentHistory, Goal, TrialRecord

__all__ = [
    "PipelineRewriter",
    "ExperimentHistory",
    "Goal",
    "TrialRecord",
]
