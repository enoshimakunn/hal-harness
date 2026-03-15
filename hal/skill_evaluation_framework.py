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
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


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


@dataclass
class SimilarityConfig:
    method: str
    embedding_model: str
    llm_model: str
    embedding_batch_size: int
    api_key: str | None


@dataclass
class SimilarityRuntime:
    config: SimilarityConfig
    client: OpenAI
    task_embeddings: list[list[float]] | None = None


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


def build_openai_client(api_key: str | None) -> OpenAI:
    if api_key:
        return OpenAI(api_key=api_key)
    return OpenAI()


def chunked(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[idx : idx + chunk_size] for idx in range(0, len(items), chunk_size)]


def embed_texts(client: OpenAI, model: str, texts: list[str], batch_size: int) -> list[list[float]]:
    if not texts:
        return []

    embeddings: list[list[float]] = []
    for batch in chunked(texts, batch_size):
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend([item.embedding for item in response.data])
    return embeddings


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Embedding vectors must have the same length.")

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b, strict=True):
        dot_product += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / ((norm_a ** 0.5) * (norm_b ** 0.5))


def llm_similarity_score(client: OpenAI, model: str, skill_description: str, task_description: str) -> float:
    if not skill_description.strip() or not task_description.strip():
        return 0.0

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You score how well a skill description matches a task description. "
                    "Return only a single floating point number between 0 and 1, where 1 means "
                    "the skill is highly relevant and 0 means it is not relevant."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Score the similarity between the following skill and task.\n\n"
                    "Scoring rubric:\n"
                    "- 0.0 to 0.2: almost no overlap in capability\n"
                    "- 0.2 to 0.5: limited or indirect relevance\n"
                    "- 0.5 to 0.8: clearly relevant\n"
                    "- 0.8 to 1.0: highly aligned and directly applicable\n\n"
                    f"Skill description:\n{skill_description.strip()}\n\n"
                    f"Task description:\n{task_description.strip()}\n\n"
                    "Output only the numeric score."
                ),
            },
        ],
        temperature=0,
        max_tokens=16,
    )

    content = response.choices[0].message.content or ""
    match = re.search(r"-?\d+(?:\.\d+)?", content)
    if not match:
        raise ValueError(f"Could not parse similarity score from model output: {content!r}")

    score = float(match.group(0))
    return max(0.0, min(1.0, score))


def build_similarity_runtime(
    config: SimilarityConfig,
    tasks: list[TaskInfo],
) -> SimilarityRuntime:
    client = build_openai_client(config.api_key)
    runtime = SimilarityRuntime(config=config, client=client)

    if config.method == "embedding":
        runtime.task_embeddings = embed_texts(
            client=client,
            model=config.embedding_model,
            texts=[task.description for task in tasks],
            batch_size=config.embedding_batch_size,
        )

    return runtime


def similarity_scores_for_skill(
    skill: SkillInfo,
    tasks: list[TaskInfo],
    runtime: SimilarityRuntime,
) -> list[float]:
    if runtime.config.method == "embedding":
        if runtime.task_embeddings is None:
            raise ValueError("Task embeddings were not initialized for embedding similarity.")
        skill_embedding = embed_texts(
            client=runtime.client,
            model=runtime.config.embedding_model,
            texts=[skill.description],
            batch_size=1,
        )[0]
        return [
            cosine_similarity(skill_embedding, task_embedding)
            for task_embedding in runtime.task_embeddings
        ]

    if runtime.config.method == "llm":
        scores: list[float] = []
        for task in tasks:
            scores.append(
                llm_similarity_score(
                    client=runtime.client,
                    model=runtime.config.llm_model,
                    skill_description=skill.description,
                    task_description=task.description,
                )
            )
        return scores

    raise ValueError(f"Unsupported similarity method: {runtime.config.method}")


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
    similarity_runtime: SimilarityRuntime,
    top_k: int,
    min_score: float | None,
) -> list[dict[str, Any]]:
    scores = similarity_scores_for_skill(skill, tasks, similarity_runtime)
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
    usaco_harness_image: str,
) -> dict[str, Any]:
    from hal.benchmark_manager import BenchmarkManager
    from hal.utils.local_runner import LocalRunner

    benchmark_manager = BenchmarkManager(agent_dir=str(agent_dir), config={})
    benchmark_obj = benchmark_manager.get_benchmark(benchmark)
    benchmark_obj.agent_args = agent_args

    full_dataset = benchmark_obj.get_dataset()
    selected_dataset = {
        task_id: full_dataset[task_id] for task_id in selected_task_ids if task_id in full_dataset
    }
    benchmark_obj.benchmark = selected_dataset

    run_dir = benchmark_obj.get_run_dir(run_id)
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

    if benchmark == "usaco":
        eval_results = evaluate_usaco_with_cached_image(
            agent_output=agent_output,
            benchmark_subset=selected_dataset,
            run_id=run_id,
            benchmark_dir=hal_root / "hal" / "benchmarks" / "USACO",
            image_name=usaco_harness_image,
        )
    else:
        eval_results = benchmark_obj.evaluate_output(agent_output, run_id)

    metrics = benchmark_obj.get_metrics(eval_results)
    return {
        **metrics,
        "raw_eval_results": eval_results,
        "raw_agent_output": agent_output,
        "wandb_enabled": False,
    }


def ensure_usaco_harness_image(image_name: str, requirements_path: Path) -> None:
    import docker

    client = docker.from_env()
    try:
        client.images.get(image_name)
        return
    except docker.errors.ImageNotFound:
        pass

    with tempfile.TemporaryDirectory(prefix="usaco_harness_build_") as tmp:
        build_dir = Path(tmp)
        shutil.copy2(requirements_path, build_dir / "requirements.txt")
        dockerfile = build_dir / "Dockerfile"
        dockerfile.write_text(
            "\n".join(
                [
                    "FROM python:3.11",
                    "WORKDIR /tmp/usaco-harness",
                    "COPY requirements.txt /tmp/usaco-harness/requirements.txt",
                    "RUN pip install --no-cache-dir -r /tmp/usaco-harness/requirements.txt",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"[hal-skill-eval] building USACO harness image: {image_name}")
        _, build_logs = client.images.build(
            path=str(build_dir),
            dockerfile="Dockerfile",
            tag=image_name,
        )
        for log in build_logs:
            if "stream" in log:
                line = log["stream"].strip()
                if line:
                    print(line)


def evaluate_usaco_with_cached_image(
    *,
    agent_output: dict[str, Any],
    benchmark_subset: dict[str, Any],
    run_id: str,
    benchmark_dir: Path,
    image_name: str,
) -> dict[str, Any]:
    import docker

    normalized_output = {}
    for task_id, task_data in agent_output.items():
        if isinstance(task_data, dict) and "answer" in task_data:
            normalized_output[task_id] = task_data["answer"]
        else:
            normalized_output[task_id] = task_data

    eval_tasks: dict[str, Any] = {}
    for task_id, response in normalized_output.items():
        if response is None:
            continue
        if not isinstance(response, str):
            response = str(response)
        if task_id not in benchmark_subset:
            continue
        eval_tasks[task_id] = {
            **benchmark_subset[task_id],
            "response": response,
        }

    if not eval_tasks:
        return {"rdict": {}, "sdict": {}, "rs": [], "ss": []}

    requirements_path = benchmark_dir / "requirements.txt"
    ensure_usaco_harness_image(image_name, requirements_path)

    client = docker.from_env()
    container = client.containers.run(
        image_name,
        command="tail -f /dev/null",
        volumes={
            str(benchmark_dir): {"bind": "/app", "mode": "rw", "chmod": "777"},
        },
        working_dir="/app",
        detach=True,
    )
    temp_file_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            json.dump(eval_tasks, temp_file)
            temp_file_path = temp_file.name

        subprocess.run(
            ["docker", "cp", temp_file_path, f"{container.id}:/app/responses_{run_id}.json"],
            check=True,
        )
        container.exec_run(
            "mkdir -p judge_sandbox/predictions/usaco judge_sandbox/solutions/usaco code_sandbox results"
        )
        cmd = (
            "python harness.py "
            f"--problem_dict_with_responses /app/responses_{run_id}.json "
            f"--run_id {run_id}"
        )
        result = container.exec_run(cmd, stream=True)
        for line in result.output:
            print(line.decode(), end="")

        rdict_result = container.exec_run(f"cat /app/results/rdict_{run_id}.json")
        sdict_result = container.exec_run(f"cat /app/results/sdict_{run_id}.json")
        rdict = json.loads(rdict_result.output.decode())
        sdict = json.loads(sdict_result.output.decode())
        return {"rdict": rdict, "sdict": sdict, "rs": list(rdict.values()), "ss": list(sdict.values())}
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        container.stop()
        container.remove()


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
        "--similarity-method",
        type=str,
        choices=["embedding", "llm"],
        default="embedding",
        help="Similarity backend used to match skills to tasks.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model used when --similarity-method=embedding.",
    )
    parser.add_argument(
        "--llm-similarity-model",
        type=str,
        default="gpt-4.1-mini",
        help="LLM used when --similarity-method=llm.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=128,
        help="Batch size for task embedding requests.",
    )
    parser.add_argument(
        "--similarity-api-key",
        type=str,
        default=None,
        help="Optional API key for similarity model calls. Falls back to environment config.",
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
        "--usaco-harness-image",
        type=str,
        default="hal-skill-eval-usaco-harness:latest",
        help="USACO harness image name used only by this skill evaluation framework.",
    )
    args = parser.parse_args()

    hal_root = Path(__file__).resolve().parent.parent
    # Mirror hal-eval CLI behavior: load API keys from repo-local .env.
    load_dotenv(hal_root / ".env")
    skills = discover_skills(args.skills_pool.resolve())
    selected_skills = select_skills(skills, args.skill, args.all_skills)
    tasks = load_benchmark_tasks(args.benchmark, hal_root)
    similarity_runtime = build_similarity_runtime(
        SimilarityConfig(
            method=args.similarity_method,
            embedding_model=args.embedding_model,
            llm_model=args.llm_similarity_model,
            embedding_batch_size=args.embedding_batch_size,
            api_key=args.similarity_api_key,
        ),
        tasks,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    summary_path = args.output_dir / f"{args.run_id_prefix}-{args.benchmark}-{timestamp}.json"

    all_rows: list[dict[str, Any]] = []
    for skill in selected_skills:
        matched_tasks = match_tasks_for_skill(
            skill=skill,
            tasks=tasks,
            similarity_runtime=similarity_runtime,
            top_k=args.top_k,
            min_score=args.min_score,
        )
        selected_task_ids = [row["task_id"] for row in matched_tasks]
        row: dict[str, Any] = {
            "skill_name": skill.name,
            "skill_md_path": str(skill.skill_md_path),
            "skill_description": skill.description,
            "similarity_method": args.similarity_method,
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
                usaco_harness_image=args.usaco_harness_image,
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
            "similarity_method": args.similarity_method,
            "embedding_model": args.embedding_model,
            "llm_similarity_model": args.llm_similarity_model,
            "embedding_batch_size": args.embedding_batch_size,
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
