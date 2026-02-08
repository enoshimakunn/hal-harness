"""
Financial Agent Benchmark for HAL Harness.

This benchmark evaluates agents on financial research and analysis tasks
from the ValsAI Finance Agent benchmark.
"""

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import litellm

from ..utils.logging_utils import print_warning, print_step
from .base_benchmark import BaseBenchmark


@dataclass
class RubricItem:
    """A single rubric criterion."""
    operator: str
    criteria: str


@dataclass
class BenchmarkQuestion:
    """A benchmark question with ground truth and rubric."""
    question: str
    answer: str
    question_type: str
    expert_time_mins: int
    rubric: list[RubricItem] = field(default_factory=list)


CORRECTNESS_PROMPT = """You are evaluating whether an AI agent's answer contains specific information.

**Criterion to check:** {criterion}

**Agent's Answer:**
{agent_answer}

Does the agent's answer contain or satisfy the criterion above?
Be lenient with:
- Numerical approximations (e.g., $95M vs $94.9M is acceptable)
- Paraphrasing (different wording of same meaning)
- Formatting differences (dates, percentages, etc.)

Respond with ONLY a JSON object:
{{"passed": true/false, "explanation": "brief reason"}}
"""

CONTRADICTION_PROMPT = """You are checking if an AI agent's answer contains contradictions to ground truth.

**Ground Truth:**
{ground_truth}

**Agent's Answer:**
{agent_answer}

Does the agent's answer directly contradict the ground truth above?
- Minor differences in wording are NOT contradictions
- Different wording of the same fact is NOT a contradiction
- Numerical approximations within 5% are NOT contradictions
- Only flag if the agent states something factually opposite to the ground truth

Respond with ONLY a JSON object:
{{"has_contradiction": true/false, "explanation": "brief reason"}}
"""

# Batch evaluation prompt for efficiency
BATCH_CORRECTNESS_PROMPT = """You are evaluating whether an AI agent's answer satisfies multiple criteria.

**Agent's Answer:**
{agent_answer}

**Criteria to check:**
{criteria_list}

For each criterion, determine if the agent's answer contains or satisfies it.
Be lenient with numerical approximations and paraphrasing.

Respond with ONLY a JSON array of objects, one per criterion:
[
  {{"criterion_id": 0, "passed": true/false, "explanation": "brief reason"}},
  {{"criterion_id": 1, "passed": true/false, "explanation": "brief reason"}},
  ...
]
"""


class FinancialAgentBenchmark(BaseBenchmark):
    """Financial Agent benchmark implementation with optimizations"""

    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "financial_agent"
        self.requires_sandbox = False
        super().__init__(agent_dir, config, requires_sandbox=self.requires_sandbox)

        # Load benchmark dataset from CSV
        csv_path = os.path.join(os.path.dirname(__file__), "financial_agent/public.csv")
        assert os.path.exists(csv_path), (
            f"Benchmark data not found at {csv_path}. "
            "Please ensure public.csv is in hal/benchmarks/financial_agent/"
        )

        self.benchmark = self._load_benchmark(csv_path)
        
        # Default configuration (will be overridden by agent_args in evaluate_output)
        self.use_batch_eval = False
        self.batch_size = 5
        self._config_initialized = False

    def _init_config_from_agent_args(self) -> None:
        """Initialize configuration options from agent_args (set via -A flags)."""
        if self._config_initialized:
            return
            
        if hasattr(self, 'agent_args') and self.agent_args:
            # Parse boolean values properly
            
            use_batch_eval = self.agent_args.get("use_batch_eval", False)
            if isinstance(use_batch_eval, str):
                use_batch_eval = use_batch_eval.lower() in ("true", "1", "yes")
            self.use_batch_eval = use_batch_eval
            
            self.batch_size = int(self.agent_args.get("batch_size", 5))
        
        self._config_initialized = True

    def _load_benchmark(self, csv_path: str) -> Dict[str, Any]:
        """Load benchmark questions and rubrics from CSV."""
        questions = {}

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                question_id = f"q_{idx}"

                # Parse rubric from string representation
                rubric_str = row.get("Rubric", "[]")
                try:
                    rubric_data = eval(rubric_str)  # CSV contains Python-style list
                    rubric = [
                        RubricItem(
                            operator=item.get("operator", "correctness"),
                            criteria=item.get("criteria", "")
                        )
                        for item in rubric_data
                    ]
                except Exception:
                    rubric = []

                questions[question_id] = {
                    "question": row.get("Question", ""),
                    "answer": row.get("Answer", ""),
                    "question_type": row.get("Question Type", ""),
                    "expert_time_mins": int(row.get("Expert time (mins)", 0)),
                    "rubric": [{"operator": r.operator, "criteria": r.criteria} for r in rubric],
                }

        return questions

    def _get_eval_model(self) -> str:
        """Get the evaluation model from agent_args or use default."""
        if hasattr(self, 'agent_args') and self.agent_args:
            return self.agent_args.get("eval_model", "gpt-4o-mini")
        return "gpt-4o-mini"



    def _evaluate_with_llm(
        self,
        prompt: str,
        eval_model: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Call LLM to evaluate a rubric item.
        
        Args:
            prompt: The formatted prompt for the LLM
            eval_model: Model name for litellm
            max_retries: Number of retries on failure
            
        Returns:
            Parsed JSON response from the LLM
        """
        
        for attempt in range(max_retries):
            try:
                response = litellm.completion(
                    model=eval_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,  # Increased for batch responses
                )
                
                content = response.choices[0].message.content.strip()
                
                # Try to parse JSON from response
                # Handle potential markdown code blocks
                if content.startswith("```"):
                    # Extract JSON from code block
                    lines = content.split("\n")
                    json_lines = []
                    in_block = False
                    for line in lines:
                        if line.startswith("```") and not in_block:
                            in_block = True
                            continue
                        elif line.startswith("```") and in_block:
                            break
                        elif in_block:
                            json_lines.append(line)
                    content = "\n".join(json_lines)
                
                result = json.loads(content)
                return result
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    continue
                print_warning(f"Failed to parse LLM response as JSON: {e}")
                print_warning(f"Raw response: {content[:200]}")
                return {"error": str(e), "raw_response": content}
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                print_warning(f"LLM evaluation failed: {e}")
                return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}

    def _evaluate_rubric_batch(
        self,
        agent_answer: str,
        correctness_items: List[Dict[str, str]],
        eval_model: str
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple correctness criteria in a single LLM call.
        
        Args:
            agent_answer: The agent's response
            correctness_items: List of correctness rubric items
            eval_model: Model to use for evaluation
            
        Returns:
            List of evaluation results
        """
        if not correctness_items:
            return []
        
        # Format criteria list
        criteria_list = "\n".join([
            f"{i}. {item['criteria']}"
            for i, item in enumerate(correctness_items)
        ])
        
        prompt = BATCH_CORRECTNESS_PROMPT.format(
            agent_answer=agent_answer,
            criteria_list=criteria_list
        )
        
        result = self._evaluate_with_llm(prompt, eval_model)
        
        # Handle errors
        if "error" in result:
            # Fallback to individual evaluation
            print_warning("Batch evaluation failed, falling back to individual evaluation")
            return []
        
        # Parse batch results
        if isinstance(result, list):
            batch_results = []
            for i, item in enumerate(correctness_items):
                if i < len(result):
                    batch_result = result[i]
                    batch_results.append({
                        "operator": "correctness",
                        "criteria": item["criteria"],
                        "passed": batch_result.get("passed", False),
                        "explanation": batch_result.get("explanation", ""),
                        "error": None
                    })
                else:
                    # Missing result for this criterion
                    batch_results.append({
                        "operator": "correctness",
                        "criteria": item["criteria"],
                        "passed": False,
                        "explanation": "Missing from batch response",
                        "error": "Missing result"
                    })
            return batch_results
        else:
            # Unexpected response format
            return []

    def _evaluate_rubric_item(
        self,
        agent_answer: str,
        ground_truth: str,
        rubric_item: Dict[str, str],
        eval_model: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single rubric item.
        
        Args:
            agent_answer: The agent's response
            ground_truth: The expected answer
            rubric_item: Dict with 'operator' and 'criteria'
            eval_model: Model to use for evaluation
            
        Returns:
            Evaluation result with passed/failed status
        """
        operator = rubric_item.get("operator", "correctness")
        criteria = rubric_item.get("criteria", "")
        
        if operator == "correctness":
            prompt = CORRECTNESS_PROMPT.format(
                criterion=criteria,
                agent_answer=agent_answer
            )
            result = self._evaluate_with_llm(prompt, eval_model)
            return {
                "operator": operator,
                "criteria": criteria,
                "passed": result.get("passed", False),
                "explanation": result.get("explanation", ""),
                "error": result.get("error")
            }
            
        elif operator == "contradiction":
            # Use actual ground_truth (full expert answer), not criteria
            prompt = CONTRADICTION_PROMPT.format(
                ground_truth=ground_truth,
                agent_answer=agent_answer
            )
            result = self._evaluate_with_llm(prompt, eval_model)
            # For contradiction, NOT having a contradiction is good (passed)
            has_contradiction = result.get("has_contradiction", False)
            return {
                "operator": operator,
                "criteria": "No contradiction with expert answer",
                "passed": not has_contradiction,
                "has_contradiction": has_contradiction,
                "explanation": result.get("explanation", ""),
                "error": result.get("error")
            }
        
        else:
            return {
                "operator": operator,
                "criteria": criteria,
                "passed": False,
                "error": f"Unknown operator: {operator}"
            }

    def evaluate_output(
        self, agent_output: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
        """Evaluate agent outputs using LLM-as-judge"""
        try:
            # Initialize config from agent_args on first call
            self._init_config_from_agent_args()
            
            # Normalize agent output
            normalized_output = self._normalize_agent_output(agent_output)

            # Get evaluation model
            eval_model = self._get_eval_model()
            print_step(f"Using evaluation model: {eval_model}")
            if self.use_batch_eval:
                print_step(f"Batch evaluation enabled (batch size: {self.batch_size})")

            # Track results
            results = {}
            missing_responses = []
            total_questions = len(normalized_output)
            processed = 0

            for question_id, response in normalized_output.items():
                processed += 1
                
                if response is None:
                    missing_responses.append(question_id)
                    continue

                if question_id not in self.benchmark:
                    print_warning(f"Unknown question ID: {question_id}")
                    continue

                benchmark_q = self.benchmark[question_id]
                agent_answer = str(response)
                ground_truth = benchmark_q["answer"]
                rubric = benchmark_q["rubric"]
                
                # Check if agent response is an error message - skip LLM evaluation
                error_prefixes = ("ERROR:", "Model exception", "Traceback (most recent call last)")
                is_error_response = any(agent_answer.startswith(prefix) for prefix in error_prefixes)
                
                if is_error_response:
                    # Skip LLM evaluation for error responses - automatically score 0
                    results[question_id] = {
                        "agent_answer": agent_answer[:500] + "..." if len(agent_answer) > 500 else agent_answer,
                        "ground_truth": ground_truth[:500] + "..." if len(ground_truth) > 500 else ground_truth,
                        "question_type": benchmark_q["question_type"],
                        "rubric_results": [],
                        "correctness_score": 0.0,
                        "correctness_passed": 0,
                        "correctness_total": len([r for r in rubric if r.get("operator") == "correctness"]),
                        "has_contradiction": False,
                        "final_score": 0.0,
                        "error": "Agent returned error response",
                    }
                    continue
                
                # Separate correctness and contradiction items
                correctness_items = [r for r in rubric if r.get("operator") == "correctness"]
                contradiction_items = [r for r in rubric if r.get("operator") == "contradiction"]
                
                # Evaluate rubric items
                rubric_results = []
                
                # Use batch evaluation for correctness items if enabled
                if self.use_batch_eval and len(correctness_items) > 1:
                    batch_results = self._evaluate_rubric_batch(
                        agent_answer=agent_answer,
                        correctness_items=correctness_items,
                        eval_model=eval_model
                    )
                    if batch_results:
                        rubric_results.extend(batch_results)
                    else:
                        # Fallback to individual evaluation
                        for item in correctness_items:
                            result = self._evaluate_rubric_item(
                                agent_answer=agent_answer,
                                ground_truth=ground_truth,
                                rubric_item=item,
                                eval_model=eval_model
                            )
                            rubric_results.append(result)
                else:
                    # Individual evaluation
                    for item in correctness_items:
                        result = self._evaluate_rubric_item(
                            agent_answer=agent_answer,
                            ground_truth=ground_truth,
                            rubric_item=item,
                            eval_model=eval_model
                        )
                        rubric_results.append(result)
                
                # Always evaluate contradiction items individually
                for item in contradiction_items:
                    result = self._evaluate_rubric_item(
                        agent_answer=agent_answer,
                        ground_truth=ground_truth,
                        rubric_item=item,
                        eval_model=eval_model
                    )
                    rubric_results.append(result)
                
                # Calculate scores
                correctness_results = [r for r in rubric_results if r["operator"] == "correctness"]
                contradiction_results = [r for r in rubric_results if r["operator"] == "contradiction"]
                
                # Correctness score: fraction of correctness items passed
                correctness_passed = sum(1 for r in correctness_results if r.get("passed", False))
                correctness_total = len(correctness_results)
                correctness_score = correctness_passed / correctness_total if correctness_total > 0 else 1.0
                
                # Contradiction check
                has_contradiction = any(r.get("has_contradiction", False) for r in contradiction_results)
                
                # Final score using paper's conjunction logic:
                # If any contradiction exists, score is 0
                final_score = 1.0 if (correctness_passed == correctness_total and not has_contradiction) else 0.0
                
                results[question_id] = {
                    "agent_answer": agent_answer[:500] + "..." if len(agent_answer) > 500 else agent_answer,
                    "ground_truth": ground_truth[:500] + "..." if len(ground_truth) > 500 else ground_truth,
                    "question_type": benchmark_q["question_type"],
                    "rubric_results": rubric_results,
                    "correctness_score": correctness_score,
                    "correctness_passed": correctness_passed,
                    "correctness_total": correctness_total,
                    "has_contradiction": has_contradiction,
                    "final_score": final_score,
                }
                
                # Progress logging every 10 questions
                if processed % 10 == 0:
                    print_step(f"Evaluated {processed}/{total_questions} questions...")

            if missing_responses:
                truncated = ", ".join(missing_responses[:10])
                if len(missing_responses) > 10:
                    truncated += f", ...(+{len(missing_responses)-10})"
                print_warning(
                    f"Financial Agent benchmark: skipping questions with no response: {truncated}"
                )
            


            return results

        except Exception as e:
            print_warning(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from evaluation results"""
        if not eval_results:
            return {
                "average_score": 0.0,
                "average_correctness": 0.0,
                "contradiction_rate": 0.0,
                "total_questions": 0,
                "by_question_type": {},
                "class_balanced_accuracy": 0.0,
            }

        final_scores = []
        correctness_scores = []
        contradictions = []
        by_type: Dict[str, List[float]] = {}

        for question_id, result in eval_results.items():
            final_score = result.get("final_score", 0.0)
            correctness_score = result.get("correctness_score", 0.0)
            has_contradiction = result.get("has_contradiction", False)
            
            final_scores.append(final_score)
            correctness_scores.append(correctness_score)
            contradictions.append(1 if has_contradiction else 0)

            q_type = result.get("question_type", "Unknown")
            if q_type not in by_type:
                by_type[q_type] = []
            by_type[q_type].append(final_score)

        # Calculate averages by type
        type_averages = {
            q_type: sum(s) / len(s) if s else 0.0
            for q_type, s in by_type.items()
        }
        
        # Calculate class-balanced accuracy (paper's primary metric)
        class_balanced_accuracy = (
            sum(type_averages.values()) / len(type_averages)
            if type_averages else 0.0
        )

        return {
            "average_score": sum(final_scores) / len(final_scores) if final_scores else 0.0,
            "average_correctness": sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0,
            "contradiction_rate": sum(contradictions) / len(contradictions) if contradictions else 0.0,
            "total_questions": len(final_scores),
            "by_question_type": type_averages,
            "class_balanced_accuracy": class_balanced_accuracy,
        }
