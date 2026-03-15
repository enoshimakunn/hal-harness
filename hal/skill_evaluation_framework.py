#!/usr/bin/env python3
"""
Evaluate HAL skills by matching SKILL.md descriptions to benchmark task descriptions.

Current benchmark support:
- usaco
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


@dataclass
class SkillInfo:
    name: str
    skill_md_path: Path
    content: str
    description: str


@dataclass
class TaskInfo:
    task_id: str
    description: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_frontmatter(content: str) -> dict[str, str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    frontmatter: dict[str, str] = {}
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            break
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        frontmatter[key.strip().lower()] = value.strip().strip('"').strip("'")
    return frontmatter


def extract_body_summary(content: str) -> str:
    # Drop frontmatter if present, then use the first non-heading paragraph.
    body = content
    if content.startswith("---\n"):
        end_idx = content.find("\n---", 4)
        if end_idx != -1:
            body = content[end_idx + 4 :]

    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    for paragraph in paragraphs:
        if paragraph.startswith("#"):
            continue
        return paragraph
    return body.strip()


def parse_skill(skill_md_path: Path) -> SkillInfo:
    content = read_text(skill_md_path)
    frontmatter = parse_frontmatter(content)
    name = frontmatter.get("name", skill_md_path.parent.name).strip()
    description = frontmatter.get("description", "").strip()
    if not description:
        description = extract_body_summary(content)
    return SkillInfo(
        name=name,
        skill_md_path=skill_md_path,
        content=content,
        description=description,
    )


def discover_skills(skills_pool_dir: Path) -> list[SkillInfo]:
    return [parse_skill(path) for path in sorted(skills_pool_dir.rglob("SKILL.md"))]


def select_skills(
    skills: list[SkillInfo],
    skill_selector: str | None,
    all_skills: bool,
) -> list[SkillInfo]:
    if not skills:
        raise ValueError(f"No SKILL.md found under {skills}.")

    if all_skills:
        return skills

    if skill_selector:
        skill_path = Path(skill_selector)
        if skill_path.exists():
            if skill_path.is_dir():
                skill_md = skill_path / "SKILL.md"
            else:
                skill_md = skill_path
            if skill_md.name != "SKILL.md" or not skill_md.exists():
                raise ValueError("--skill path must be SKILL.md or a folder containing it.")
            return [parse_skill(skill_md)]

        matched = [
            skill
            for skill in skills
            if skill.name == skill_selector or skill.skill_md_path.parent.name == skill_selector
        ]
        if len(matched) == 1:
            return matched
        if len(matched) > 1:
            names = ", ".join(str(s.skill_md_path) for s in matched)
            raise ValueError(f"--skill matches multiple skills: {names}")
        raise ValueError(f"Skill not found: {skill_selector}")

    if len(skills) == 1:
        return skills

    sample = ", ".join(skill.name for skill in skills[:10])
    raise ValueError(
        f"Found {len(skills)} skills. Use --skill <name_or_path> or --all-skills. Sample: {sample}"
    )


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    tokenized_docs = [tokenize(doc) for doc in docs]
    tokenized_query = tokenize(query)
    if not tokenized_docs or not tokenized_query:
        return [0.0 for _ in docs]

    doc_freq: Counter[str] = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    doc_term_freqs = [Counter(tokens) for tokens in tokenized_docs]
    avg_doc_len = sum(len(tokens) for tokens in tokenized_docs) / len(tokenized_docs)
    if avg_doc_len == 0:
        return [0.0 for _ in docs]

    n_docs = len(tokenized_docs)
    scores: list[float] = []
    for term_freqs, tokens in zip(doc_term_freqs, tokenized_docs, strict=True):
        score = 0.0
        doc_len = len(tokens)
        for term in tokenized_query:
            tf = term_freqs.get(term, 0)
            if tf == 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
            score += idf * (tf * (k1 + 1.0) / denom)
        scores.append(score)
    return scores


def load_benchmark_tasks(benchmark_name: str, hal_dir: Path) -> list[TaskInfo]:
    if benchmark_name != "usaco":
        raise ValueError(f"Unsupported benchmark '{benchmark_name}'. Currently supported: usaco")

    dataset_path = (
        hal_dir / "hal" / "benchmarks" / "USACO" / "data" / "datasets" / "usaco_subset307_dict.json"
    )
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"USACO dataset not found at {dataset_path}. Download and place benchmark data first."
        )

    data = json.loads(read_text(dataset_path))
    tasks: list[TaskInfo] = []
    for task_id, task in data.items():
        description = task.get("description_no_samples") or task.get("description") or ""
        tasks.append(TaskInfo(task_id=task_id, description=description))
    return tasks


def sanitize_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()


def parse_agent_args(raw_args: list[str]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for raw in raw_args:
        if "=" not in raw:
            raise ValueError(f"Invalid --agent-arg '{raw}'. Expected key=value.")
        key, value = raw.split("=", 1)
        value = value.strip()

        # best-effort parsing for bool/int/float/json values
        lowered = value.lower()
        if lowered in {"true", "false"}:
            parsed: Any = lowered == "true"
        else:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                try:
                    parsed = int(value)
                except ValueError:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = value
        args[key.strip().replace("-", "_")] = parsed
    return args


def match_tasks_for_skill(
    skill: SkillInfo,
    tasks: list[TaskInfo],
    top_k: int,
    min_score: float | None,
) -> list[dict[str, Any]]:
    scores = bm25_scores(skill.description, [task.description for task in tasks])
    rows = [
        {
            "task_id": task.task_id,
            "score": float(score),
            "description_preview": task.description[:240],
        }
        for task, score in zip(tasks, scores, strict=True)
    ]
    rows.sort(key=lambda row: row["score"], reverse=True)
    if min_score is not None:
        rows = [row for row in rows if row["score"] >= min_score]
    return rows[:top_k]


def run_skill_eval(
    *,
    benchmark: str,
    run_id: str,
    agent_dir: Path,
    agent_function: str,
    agent_name: str,
    max_concurrent: int,
    agent_args: dict[str, Any],
    selected_task_ids: list[str],
    hal_root: Path,
    docker: bool = False,
) -> dict[str, Any]:
    from hal.benchmark_manager import BenchmarkManager
    from hal.utils.local_runner import LocalRunner
    from hal.utils.docker_runner import DockerRunner

    benchmark_manager = BenchmarkManager(agent_dir=str(agent_dir), config={})
    benchmark_obj = benchmark_manager.get_benchmark(benchmark)
    benchmark_obj.agent_args = agent_args

    full_dataset = benchmark_obj.get_dataset()
    selected_dataset = {
        task_id: full_dataset[task_id] for task_id in selected_task_ids if task_id in full_dataset
    }
    benchmark_obj.benchmark = selected_dataset

    run_dir = benchmark_obj.get_run_dir(run_id)
    if docker:
        runner = DockerRunner(
            log_dir=run_dir,
            max_concurrent=max_concurrent,
            benchmark=benchmark_obj,
        )
    else:
        runner = LocalRunner(
            log_dir=run_dir,
            max_concurrent=max_concurrent,
            conda_env=None,
            benchmark=benchmark_obj,
        )
    agent_output = asyncio.run(
        runner.run_agent(
            dataset=selected_dataset,
            agent_function=agent_function,
            agent_dir=str(agent_dir),
            agent_args=agent_args,
            run_id=run_id,
            benchmark=benchmark_obj,
        )
    )

    eval_results = benchmark_obj.evaluate_output(agent_output, run_id)

    metrics = benchmark_obj.get_metrics(eval_results)
    return {
        **metrics,
        "raw_eval_results": eval_results,
        "raw_agent_output": agent_output,
        "wandb_enabled": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Match SKILL.md descriptions against benchmark tasks and optionally run HAL evaluation."
        )
    )
    parser.add_argument(
        "--skills-pool",
        type=Path,
        default=Path("hal/skills-pool"),
        help="Directory containing skills with SKILL.md files.",
    )
    parser.add_argument(
        "--skill",
        type=str,
        default=None,
        help="Skill name, skill directory path, or SKILL.md path.",
    )
    parser.add_argument(
        "--all-skills",
        action="store_true",
        help="Evaluate all skills found under --skills-pool.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="usaco",
        help="Benchmark name. Currently supports: usaco.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Top matched tasks per skill.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional score floor; tasks below this are dropped before top-k.",
    )
    parser.add_argument(
        "--agent-dir",
        type=Path,
        default=Path("agents/usaco_example_agent"),
        help="Agent directory used by HAL runner.",
    )
    parser.add_argument(
        "--agent-function",
        type=str,
        default="main.run",
        help="Agent function in module.function format.",
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        default="USACO Example Agent (Skill Eval)",
        help="Agent name for HAL results.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name passed as agent arg model_name.",
    )
    parser.add_argument(
        "--agent-arg",
        action="append",
        default=[],
        help="Additional agent arg in key=value form. Can be repeated.",
    )
    parser.add_argument(
        "--skill-agent-arg",
        type=str,
        default="skill_content",
        help="Agent kwarg key used to pass selected SKILL.md content.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run HAL evaluation after matching. Without this flag, only task matching is produced.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Concurrency for HAL task execution.",
    )
    parser.add_argument(
        "--run-id-prefix",
        type=str,
        default="hal-skill-eval",
        help="Prefix for generated HAL run_id values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/skill_evaluation"),
        help="Directory where summary JSON files are written.",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help=(
            "Run the agent inside Docker containers (similar to hal-eval --docker). "
            "Evaluation still follows each benchmark's own logic."
        ),
    )
    args = parser.parse_args()

    hal_root = Path(__file__).resolve().parent.parent
    # Mirror hal-eval CLI behavior: load API keys from repo-local .env.
    load_dotenv(hal_root / ".env")
    skills = discover_skills(args.skills_pool.resolve())
    selected_skills = select_skills(skills, args.skill, args.all_skills)
    tasks = load_benchmark_tasks(args.benchmark, hal_root)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    summary_path = args.output_dir / f"{args.run_id_prefix}-{args.benchmark}-{timestamp}.json"

    all_rows: list[dict[str, Any]] = []
    for skill in selected_skills:
        matched_tasks = match_tasks_for_skill(
            skill=skill,
            tasks=tasks,
            top_k=args.top_k,
            min_score=args.min_score,
        )
        selected_task_ids = [row["task_id"] for row in matched_tasks]
        row: dict[str, Any] = {
            "skill_name": skill.name,
            "skill_md_path": str(skill.skill_md_path),
            "skill_description": skill.description,
            "n_matched_tasks": len(matched_tasks),
            "matched_tasks": matched_tasks,
            "evaluation": None,
        }

        if args.run and selected_task_ids:
            run_id = f"{sanitize_slug(args.run_id_prefix)}-{sanitize_slug(skill.name)}-{timestamp}"
            run_agent_args = parse_agent_args(args.agent_arg)
            run_agent_args["benchmark_name"] = args.benchmark
            if args.model_name:
                run_agent_args["model_name"] = args.model_name
            run_agent_args[args.skill_agent_arg] = skill.content

            eval_results = run_skill_eval(
                benchmark=args.benchmark,
                run_id=run_id,
                agent_dir=args.agent_dir.resolve(),
                agent_function=args.agent_function,
                agent_name=args.agent_name,
                max_concurrent=args.max_concurrent,
                agent_args=run_agent_args,
                selected_task_ids=selected_task_ids,
                hal_root=hal_root,
                docker=args.docker,
            )
            row["evaluation"] = {
                "run_id": run_id,
                "benchmark": args.benchmark,
                "agent_dir": str(args.agent_dir),
                "agent_function": args.agent_function,
                "agent_name": args.agent_name,
                "n_selected_tasks": len(selected_task_ids),
                "results": eval_results,
            }

        all_rows.append(row)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "skills_pool": str(args.skills_pool),
            "skill": args.skill,
            "all_skills": args.all_skills,
            "benchmark": args.benchmark,
            "top_k": args.top_k,
            "min_score": args.min_score,
            "agent_dir": str(args.agent_dir),
            "agent_function": args.agent_function,
            "agent_name": args.agent_name,
            "model_name": args.model_name,
            "run": args.run,
            "max_concurrent": args.max_concurrent,
        },
        "skills": all_rows,
    }
    write_json(summary_path, summary)

    print(f"[hal-skill-eval] summary: {summary_path}")
    for row in all_rows:
        print(
            f"[hal-skill-eval] skill={row['skill_name']} matched={row['n_matched_tasks']} "
            f"top_task={(row['matched_tasks'][0]['task_id'] if row['matched_tasks'] else 'none')}"
        )


if __name__ == "__main__":
    main()
