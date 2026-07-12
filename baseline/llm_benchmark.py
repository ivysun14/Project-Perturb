"""
LLM Benchmark for Perturbation Prediction

Benchmarks LLMs on predicting gene expression changes after perturbation
using step-by-step reasoning.

Uses OpenRouter API to access models (e.g., Claude Sonnet 4.5).

Dependencies:
    pip install requests tqdm scikit-learn numpy

Usage:
    # Set API key
    export OPENROUTER_API_KEY="your-key-here"

    # Test with 5 samples on Adamson dataset (default)
    python llm_benchmark.py --max_samples 5

    # Use different dataset
    python llm_benchmark.py --dataset adamson --max_samples 5
    python llm_benchmark.py --dataset norman --max_samples 5

    # Custom JSONL file
    python llm_benchmark.py --test_data /path/to/your.jsonl --max_samples 5

Available datasets (in ../output/):
    - adamson: CRISPRi in K562 cells (default)
    - Add more by placing {name}_direction_testing.jsonl in ../output/
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import requests
from tqdm import tqdm

# Add parent directory for imports
import sys
sys.path.insert(0, '..')
from perturbseq_lang.evaluation import evaluate_three_class, print_three_class_report


# =============================================================================
# OpenRouter Client
# =============================================================================

@dataclass
class OpenRouterClient:
    """Client for OpenRouter API to access multiple LLMs."""

    api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"

    # Model mappings
    MODEL_IDS = {
        "claude-sonnet-4.5": "anthropic/claude-sonnet-4.5",
    }

    def __post_init__(self):
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Get your key at https://openrouter.ai/keys"
            )

    def chat(
        self,
        model: str,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        retry_count: int = 3,
        retry_delay: float = 2.0,
    ) -> Dict:
        """
        Send a chat completion request to OpenRouter.

        Args:
            model: Model name (e.g., "gpt-4o", "claude-sonnet-4.5")
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens in response
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds

        Returns:
            Response dict with 'content' and 'usage' keys
        """
        model_id = self.MODEL_IDS.get(model, model)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/perturbseq-lang",
            "X-Title": "PerturbSeq LLM Benchmark"
        }

        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(retry_count):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()

                return {
                    "content": data["choices"][0]["message"]["content"],
                    "usage": data.get("usage", {}),
                    "model": data.get("model", model_id),
                }

            except requests.exceptions.RequestException as e:
                if attempt < retry_count - 1:
                    print(f"  Request failed (attempt {attempt + 1}): {e}")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"Failed after {retry_count} attempts: {e}")

        return {}


# =============================================================================
# Prompt Templates
# =============================================================================

SYSTEM_PROMPT = """You are an expert computational biologist. You will be given training examples showing how perturbing different genes affects downstream gene expression in a Perturb-seq experiment.

Use these training examples to learn patterns and predict the effect of a NEW unseen perturbation.

Always respond with valid JSON only."""


def format_training_example(example: Dict) -> str:
    """Format a single training example for the prompt."""
    problem = example["problem"]
    answer = example["answer"]

    # Extract perturbed gene
    perturbed = extract_perturbed_gene(problem)

    # Format the answer
    inc = answer.get("INCREASE", [])
    dec = answer.get("DECREASE", [])
    nc = answer.get("NO_CHANGE", [])

    parts = []
    if inc:
        parts.append(f"INCREASE: {', '.join(inc[:10])}" + ("..." if len(inc) > 10 else ""))
    if dec:
        parts.append(f"DECREASE: {', '.join(dec[:10])}" + ("..." if len(dec) > 10 else ""))
    if nc:
        parts.append(f"NO_CHANGE: {', '.join(nc[:5])}" + ("..." if len(nc) > 5 else ""))

    return f"Perturbing {perturbed} → {'; '.join(parts)}"


def create_fewshot_prompt(
    test_example: Dict,
    training_examples: List[Dict],
    n_shots: int = 5
) -> str:
    """
    Create a few-shot prompt with training examples.

    Args:
        test_example: The test example to predict
        training_examples: List of training examples to use as context
        n_shots: Number of training examples to include
    """
    # Sample training examples (could be random or selected)
    import random
    shots = random.sample(training_examples, min(n_shots, len(training_examples)))

    # Format training examples
    training_text = "\n".join([f"  - {format_training_example(ex)}" for ex in shots])

    # Extract info from test example
    test_problem = test_example["problem"]
    perturbed_gene = extract_perturbed_gene(test_problem)
    target_genes = extract_genes_from_problem(test_problem)

    prompt = f"""Here are training examples from a Perturb-seq experiment showing how gene perturbations affect downstream genes:

{training_text}

Now predict the effect of a NEW perturbation:

**Perturbed gene:** {perturbed_gene}
**Target genes to predict:** {', '.join(target_genes)}

Based on the patterns in the training examples and your biological knowledge, predict whether each target gene will INCREASE, DECREASE, or show NO_CHANGE.

Think step-by-step:
1. What is the function of {perturbed_gene}?
2. Are any target genes related to genes affected in the training examples?
3. What patterns do you see in the training data?

Respond with JSON:
```json
{{
  "reasoning": "Brief explanation of your prediction logic",
  "predictions": {{
    "INCREASE": ["gene1", "gene2"],
    "DECREASE": ["gene3"],
    "NO_CHANGE": ["gene4", "gene5"]
  }}
}}
```"""

    return prompt


def create_cot_prompt(problem: str) -> str:
    """Create a chain-of-thought reasoning prompt (no training examples)."""
    return f"""{problem}

Think through this step-by-step:

1. **Perturbed Gene Analysis**: What is the function of the perturbed gene? What pathways is it involved in?

2. **Target Gene Analysis**: For each target gene, consider:
   - Is it directly regulated by the perturbed gene?
   - Is it in the same pathway?
   - What is its known function?

3. **Prediction Reasoning**: Based on the above, predict the direction of change for each gene.

Respond with a JSON object containing:
- "reasoning": A brief explanation of your key predictions (2-3 sentences)
- "predictions": {{"INCREASE": [...], "DECREASE": [...], "NO_CHANGE": [...]}}

Example format:
```json
{{
  "reasoning": "Gene X is a transcription factor that activates genes A, B, C. Knocking it down should decrease their expression. Genes D, E are in a compensatory pathway that may increase.",
  "predictions": {{
    "INCREASE": ["D", "E"],
    "DECREASE": ["A", "B", "C"],
    "NO_CHANGE": ["F", "G"]
  }}
}}
```

Now analyze the perturbation and provide your prediction:"""


def create_simple_prompt(problem: str) -> str:
    """Create a simpler direct prediction prompt."""
    return f"""{problem}

Analyze this perturbation experiment and predict the expression changes.

Respond ONLY with a JSON object in this exact format:
```json
{{
  "reasoning": "Brief explanation of your prediction logic",
  "predictions": {{
    "INCREASE": ["gene1", "gene2"],
    "DECREASE": ["gene3", "gene4"],
    "NO_CHANGE": ["gene5"]
  }}
}}
```

Your prediction:"""


# =============================================================================
# Data Loading
# =============================================================================

def load_jsonl(path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_genes_from_problem(problem: str) -> List[str]:
    """Extract the list of target genes from the problem text."""
    # Find the gene list in the problem (after "following genes:")
    import re
    match = re.search(r'following genes:\s*([^?]+)\?', problem)
    if match:
        genes_str = match.group(1)
        # Split by comma and clean up
        genes = [g.strip() for g in genes_str.split(',')]
        return genes
    return []


def extract_perturbed_gene(problem: str) -> str:
    """Extract the perturbed gene name from the problem text."""
    import re
    match = re.search(r'Perturbing gene (\w+)', problem)
    if match:
        return match.group(1)
    return ""


# =============================================================================
# Prediction and Evaluation
# =============================================================================

def parse_llm_response(response: str, expected_genes: List[str]) -> Dict[str, List[str]]:
    """
    Parse LLM response to extract predictions.

    Args:
        response: Raw LLM response text
        expected_genes: List of genes we expect predictions for

    Returns:
        Dict with INCREASE, DECREASE, NO_CHANGE lists
    """
    import re

    # Try to extract JSON from response
    # Handle markdown code blocks
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON object
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Failed to parse - return empty predictions
            return {"INCREASE": [], "DECREASE": [], "NO_CHANGE": expected_genes}

    try:
        parsed = json.loads(json_str)

        # Handle nested structure with "predictions" key
        if "predictions" in parsed:
            predictions = parsed["predictions"]
        else:
            predictions = parsed

        # Normalize keys and extract gene lists
        result = {
            "INCREASE": [],
            "DECREASE": [],
            "NO_CHANGE": []
        }

        for key, genes in predictions.items():
            key_upper = key.upper()
            if key_upper in result:
                if isinstance(genes, list):
                    result[key_upper] = [str(g) for g in genes]

        # Validate that all expected genes are accounted for
        all_predicted = set(result["INCREASE"]) | set(result["DECREASE"]) | set(result["NO_CHANGE"])
        missing = set(expected_genes) - all_predicted

        # Add missing genes to NO_CHANGE
        result["NO_CHANGE"].extend(list(missing))

        return result

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"  Warning: Failed to parse response: {e}")
        return {"INCREASE": [], "DECREASE": [], "NO_CHANGE": expected_genes}


def flatten_predictions(predictions: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """Convert predictions dict to list of (gene, class) tuples."""
    result = []
    for cls, genes in predictions.items():
        for gene in genes:
            result.append((gene, cls.lower()))
    return result


def evaluate_predictions(
    all_predictions: List[Dict[str, List[str]]],
    all_ground_truth: List[Dict[str, List[str]]],
) -> Dict:
    """
    Evaluate predictions against ground truth.

    Args:
        all_predictions: List of prediction dicts per example
        all_ground_truth: List of ground truth dicts per example

    Returns:
        Evaluation metrics dict
    """
    # Flatten to per-gene predictions
    pred_flat = []
    true_flat = []

    for pred, truth in zip(all_predictions, all_ground_truth):
        # Get all genes from ground truth
        for cls in ["INCREASE", "DECREASE", "NO_CHANGE"]:
            for gene in truth.get(cls, []):
                # Find this gene's predicted class
                pred_cls = "NO_CHANGE"  # default
                for p_cls in ["INCREASE", "DECREASE", "NO_CHANGE"]:
                    if gene in pred.get(p_cls, []):
                        pred_cls = p_cls
                        break

                pred_flat.append(pred_cls.lower().replace("_", "_"))
                true_flat.append(cls.lower().replace("_", "_"))

    # Map to standard labels
    label_map = {
        "increase": "up",
        "decrease": "down",
        "no_change": "not_de"
    }

    pred_mapped = [label_map.get(p, p) for p in pred_flat]
    true_mapped = [label_map.get(t, t) for t in true_flat]

    return evaluate_three_class(pred_mapped, true_mapped, labels=['down', 'not_de', 'up'])


def run_benchmark(
    client: OpenRouterClient,
    model: str,
    test_data: List[Dict],
    training_data: Optional[List[Dict]] = None,
    prompt_style: str = "fewshot",
    n_shots: int = 5,
    max_samples: Optional[int] = None,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Run the LLM benchmark on test data.

    Args:
        client: OpenRouter client
        model: Model name to use
        test_data: List of test examples from JSONL
        training_data: List of training examples for few-shot prompts
        prompt_style: "fewshot", "cot", or "simple"
        n_shots: Number of training examples to include in few-shot prompt
        max_samples: Maximum number of samples to evaluate (None = all)
        save_path: Path to save detailed results

    Returns:
        Evaluation results dict
    """
    if max_samples:
        test_data = test_data[:max_samples]

    print(f"\nRunning benchmark: {model}")
    print(f"  Samples: {len(test_data)}")
    print(f"  Prompt style: {prompt_style}")
    if prompt_style == "fewshot":
        print(f"  N-shots: {n_shots}")
        if not training_data:
            raise ValueError("Training data required for fewshot prompt style")

    all_predictions = []
    all_ground_truth = []
    detailed_results = []

    for i, example in enumerate(tqdm(test_data, desc=f"  {model}")):
        problem = example["problem"]
        ground_truth = example["answer"]

        # Extract expected genes
        expected_genes = extract_genes_from_problem(problem)
        perturbed_gene = extract_perturbed_gene(problem)

        # Create prompt based on style
        if prompt_style == "fewshot":
            user_prompt = create_fewshot_prompt(example, training_data, n_shots=n_shots)
        elif prompt_style == "cot":
            user_prompt = create_cot_prompt(problem)
        else:
            user_prompt = create_simple_prompt(problem)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = client.chat(model=model, messages=messages)
            content = response["content"]

            # Parse predictions
            predictions = parse_llm_response(content, expected_genes)

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            predictions = {"INCREASE": [], "DECREASE": [], "NO_CHANGE": expected_genes}
            content = f"ERROR: {e}"

        all_predictions.append(predictions)
        all_ground_truth.append(ground_truth)

        # Store detailed result (including full prompt for debugging)
        detailed_results.append({
            "index": i,
            "perturbed_gene": perturbed_gene,
            "target_genes": expected_genes,
            "prompt": user_prompt,
            "ground_truth": ground_truth,
            "predictions": predictions,
            "raw_response": content[:2000],  # Store more for debugging
        })

        # Rate limiting
        time.sleep(0.5)

    # Evaluate
    results = evaluate_predictions(all_predictions, all_ground_truth)
    results["model"] = model
    results["n_samples"] = len(test_data)
    results["prompt_style"] = prompt_style
    results["n_shots"] = n_shots if prompt_style == "fewshot" else 0

    # Save detailed results
    if save_path:
        output = {
            "model": model,
            "prompt_style": prompt_style,
            "n_shots": n_shots if prompt_style == "fewshot" else 0,
            "metrics": results,
            "samples": detailed_results
        }
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"  Saved detailed results to {save_path}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM Benchmark for Perturbation Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python llm_benchmark.py --max_samples 5                    # Test 5 samples
    python llm_benchmark.py --dataset adamson --max_samples 10 # Adamson dataset
    python llm_benchmark.py --test_data custom.jsonl           # Custom file
        """
    )
    parser.add_argument("--dataset", type=str, default="adamson",
                        help="Dataset name (looks for {name}_direction_testing.jsonl in ../output/)")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Override: direct path to test JSONL file")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["claude-sonnet-4.5"],
                        help="Models to benchmark")
    parser.add_argument("--prompt_style", type=str, choices=["fewshot", "cot", "simple"],
                        default="fewshot", help="Prompt style: fewshot=with training examples, cot=chain-of-thought, simple=direct")
    parser.add_argument("--n_shots", type=str, default="all",
                        help="Number of training examples: integer or 'all' (default: all)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (for testing)")
    parser.add_argument("--output_dir", type=str, default="../output/llm_benchmark",
                        help="Output directory for results")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine test data path
    if args.test_data:
        test_data_path = args.test_data
    else:
        test_data_path = f"../output/{args.dataset}_direction_testing.jsonl"

    # Load test data
    print(f"Loading test data from {test_data_path}")
    if not Path(test_data_path).exists():
        print(f"ERROR: File not found: {test_data_path}")
        print(f"Available datasets in ../output/:")
        for f in Path("../output").glob("*_direction_testing.jsonl"):
            print(f"  - {f.stem.replace('_direction_testing', '')}")
        sys.exit(1)

    test_data = load_jsonl(test_data_path)
    print(f"Loaded {len(test_data)} test examples")

    # Parse n_shots
    if args.n_shots.lower() == "all":
        n_shots = None  # Will use all training examples
    else:
        n_shots = int(args.n_shots)

    # Load training data if using few-shot
    training_data = None
    if args.prompt_style == "fewshot":
        if args.test_data:
            train_data_path = args.test_data.replace("_testing", "_training")
        else:
            train_data_path = f"../output/{args.dataset}_direction_training.jsonl"

        if Path(train_data_path).exists():
            training_data = load_jsonl(train_data_path)
            print(f"Loaded {len(training_data)} training examples for few-shot")
            if n_shots is None:
                n_shots = len(training_data)
                print(f"Using ALL {n_shots} training examples")
        else:
            print(f"WARNING: Training data not found at {train_data_path}")
            print("Falling back to 'cot' prompt style")
            args.prompt_style = "cot"

    # Initialize client
    client = OpenRouterClient()

    # Run benchmarks
    all_results = []

    for model in args.models:
        save_path = output_dir / f"{model.replace('/', '_')}_{args.prompt_style}_results.json"

        try:
            results = run_benchmark(
                client=client,
                model=model,
                test_data=test_data,
                training_data=training_data,
                prompt_style=args.prompt_style,
                n_shots=n_shots,
                max_samples=args.max_samples,
                save_path=str(save_path)
            )
            all_results.append(results)

            print(f"\n  Results for {model}:")
            print(f"    Accuracy: {results['accuracy']:.4f}")
            print(f"    Macro F1: {results['macro_avg']['f1']:.4f}")

        except Exception as e:
            print(f"  Failed to benchmark {model}: {e}")

    # Summary table
    if all_results:
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Model':<25} {'Accuracy':>10} {'Up F1':>10} {'Down F1':>10} {'NotDE F1':>10}")
        print("-" * 70)

        for r in all_results:
            up_f1 = r['per_class'].get('up', {}).get('f1', 0)
            down_f1 = r['per_class'].get('down', {}).get('f1', 0)
            notde_f1 = r['per_class'].get('not_de', {}).get('f1', 0)
            print(f"{r['model']:<25} {r['accuracy']:>10.4f} {up_f1:>10.4f} {down_f1:>10.4f} {notde_f1:>10.4f}")

        print("=" * 70)

        # Save summary
        summary_path = output_dir / "benchmark_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
