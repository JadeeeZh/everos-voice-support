"""Run one tau2 telecom call with a given agent (power-user path, no registry)."""

from __future__ import annotations

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner import build_environment, build_user, get_tasks, run_simulation

from voicedemo.agent import ExperienceAgent, ExperienceProvider
from voicedemo.config import DOMAIN, RunConfig


def load_split(split: str, *, limit: int | None = None) -> list[Task]:
    return get_tasks(DOMAIN, task_split_name=split, num_tasks=limit)


def run_call(
    cfg: RunConfig,
    task: Task,
    *,
    experience_provider: ExperienceProvider | None = None,
) -> tuple[SimulationRun, ExperienceAgent | LLMAgent]:
    """Run one simulated support call and evaluate it.

    With ``experience_provider=None`` this is the exact baseline agent;
    otherwise the ExperienceAgent (baseline + one retrieval) is used.
    """
    env = build_environment(DOMAIN)
    if experience_provider is None:
        agent: LLMAgent = LLMAgent(
            tools=env.get_tools(),
            domain_policy=env.get_policy(),
            llm=cfg.agent_llm,
            llm_args=dict(cfg.agent_llm_args),
        )
    else:
        agent = ExperienceAgent(
            tools=env.get_tools(),
            domain_policy=env.get_policy(),
            llm=cfg.agent_llm,
            llm_args=dict(cfg.agent_llm_args),
            experience_provider=experience_provider,
            midcall_retrieval_user_turn=cfg.midcall_retrieval_user_turn,
        )
    user = build_user(
        "user_simulator",
        env,
        task,
        llm=cfg.user_llm,
        llm_args=dict(cfg.user_llm_args),
    )
    orchestrator = Orchestrator(
        domain=DOMAIN,
        agent=agent,
        user=user,
        environment=env,
        task=task,
        max_steps=cfg.max_steps,
        max_errors=cfg.max_errors,
        seed=cfg.seed,
    )
    sim = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)
    return sim, agent
