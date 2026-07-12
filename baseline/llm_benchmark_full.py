"""
LLM Benchmark with Full Gene Set

Uses perturbevolve's data_utils to get proper class distribution:
- ~12% UP, ~10% DOWN, ~78% NOT_DE

Compares prompting strategies:
- Zero-shot (no examples)
- Few-shot with N random examples
- Few-shot with ALL examples

Usage:
    export OPENROUTER_API_KEY="your-key"

    # Compare all three: 0 vs 10 vs all
    python llm_benchmark_full.py --max_samples 5

    # Just zero-shot
    python llm_benchmark_full.py --n_shots 0 --max_samples 5

    # 10 random examples
    python llm_benchmark_full.py --n_shots 10 --max_samples 5

    # 20 random examples
    python llm_benchmark_full.py --n_shots 20 --max_samples 5

    # All training examples
    python llm_benchmark_full.py --n_shots all --max_samples 5
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import requests
from tqdm import tqdm

# Add perturbevolve to path
sys.path.insert(0, '/Users/yiquntchen/Desktop/chen-lab/perturbevovle/problems/perturbation_prediction')
from data_utils import load_and_prepare_data

# Add parent for evaluation
sys.path.insert(0, '..')
from perturbseq_lang.evaluation import evaluate_three_class


# =============================================================================
# OpenRouter Client
# =============================================================================

@dataclass
class OpenRouterClient:
    """Client for OpenRouter API."""

    api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"

    MODEL_IDS = {
        "claude-sonnet-4.5": "anthropic/claude-sonnet-4.5",
    }

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Get key at https://openrouter.ai/keys")

    def chat(self, model: str, messages: List[Dict], temperature: float = 0.0,
             max_tokens: int = 4096, retry_count: int = 3) -> Dict:
        model_id = self.MODEL_IDS.get(model, model)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(retry_count):
            try:
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()
                return {"content": data["choices"][0]["message"]["content"]}
            except Exception as e:
                if attempt < retry_count - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    raise RuntimeError(f"Failed: {e}")
        return {}


# =============================================================================
# Prompt Templates
# =============================================================================

SYSTEM_PROMPT_ZEROSHOT = """You are an expert computational biologist. Predict how gene perturbations affect downstream gene expression.

For each gene, predict: UP (upregulated), DOWN (downregulated), or NOT_DE (not differentially expressed).

Respond with JSON only."""

SYSTEM_PROMPT_FEWSHOT = """You are an expert computational biologist. You will see training examples of perturbation effects, then predict for a new case.

For each gene, predict: UP, DOWN, or NOT_DE.

Respond with JSON only."""


def create_zeroshot_prompt(pert: str, genes: List[str], cell_type: str = "K562") -> str:
    """Zero-shot: no training examples, just biological knowledge."""
    genes_str = ", ".join(genes)

    return f"""Predict the effect of knocking down gene {pert} on these genes in {cell_type} cells:

Target genes: {genes_str}

Based on your biological knowledge:
1. What does {pert} do?
2. How might its knockdown affect each target gene?

Respond with JSON:
```json
{{
  "reasoning": "Brief explanation",
  "predictions": {{
    "UP": ["gene1", ...],
    "DOWN": ["gene2", ...],
    "NOT_DE": ["gene3", ...]
  }}
}}
```"""


def create_fewshot_prompt(pert: str, genes: List[str], training_examples: List[Dict],
                          n_shots: int = None, cell_type: str = "K562") -> str:
    """Few-shot: with training examples. n_shots=None means use all."""

    # Select examples
    if n_shots is None or n_shots >= len(training_examples):
        shots = training_examples
    else:
        shots = random.sample(training_examples, n_shots)

    # Format training examples compactly
    train_text = []
    for ex in shots:
        train_pert = ex['pert']
        up = [g for g, l in zip(ex['genes'], ex['labels']) if l == 'up'][:5]
        down = [g for g, l in zip(ex['genes'], ex['labels']) if l == 'down'][:5]
        not_de = [g for g, l in zip(ex['genes'], ex['labels']) if l == 'not_de'][:3]

        parts = []
        if up:
            parts.append(f"UP: {', '.join(up)}")
        if down:
            parts.append(f"DOWN: {', '.join(down)}")
        if not_de:
            parts.append(f"NOT_DE: {', '.join(not_de)}...")

        train_text.append(f"  - {train_pert} → {'; '.join(parts)}")

    genes_str = ", ".join(genes)

    return f"""Training examples from {cell_type} Perturb-seq ({len(shots)} perturbations):

{chr(10).join(train_text)}

---
NEW perturbation to predict:

**Perturbed gene:** {pert}
**Target genes:** {genes_str}

Predict UP, DOWN, or NOT_DE for each gene.

```json
{{
  "reasoning": "Brief explanation",
  "predictions": {{
    "UP": ["gene1", ...],
    "DOWN": ["gene2", ...],
    "NOT_DE": ["gene3", ...]
  }}
}}
```"""


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_test_samples(data: Dict, max_samples: int = 100, genes_per_sample: int = 20) -> List[Dict]:
    """Prepare test samples (one per perturbation)."""
    test_keys = data['test']['keys']
    test_labels = data['test']['labels']

    # Group by perturbation
    pert_to_data = {}
    for (pert, gene), label in zip(test_keys, test_labels):
        if pert not in pert_to_data:
            pert_to_data[pert] = {'genes': [], 'labels': []}
        pert_to_data[pert]['genes'].append(gene)
        pert_to_data[pert]['labels'].append(label)

    # Create samples
    samples = []
    for pert, pdata in pert_to_data.items():
        genes = pdata['genes']
        labels = pdata['labels']

        # Sample genes if needed
        if len(genes) > genes_per_sample:
            indices = random.sample(range(len(genes)), genes_per_sample)
            genes = [genes[i] for i in indices]
            labels = [labels[i] for i in indices]

        samples.append({'pert': pert, 'genes': genes, 'labels': labels})

        if len(samples) >= max_samples:
            break

    return samples


def prepare_training_examples(data: Dict) -> List[Dict]:
    """Prepare ALL training examples (one per perturbation)."""
    train_keys = data['train']['keys']
    train_labels = data['train']['labels']

    pert_to_data = {}
    for (pert, gene), label in zip(train_keys, train_labels):
        if pert not in pert_to_data:
            pert_to_data[pert] = {'genes': [], 'labels': []}
        pert_to_data[pert]['genes'].append(gene)
        pert_to_data[pert]['labels'].append(label)

    return [{'pert': p, 'genes': d['genes'], 'labels': d['labels']}
            for p, d in pert_to_data.items()]


# =============================================================================
# Parsing and Evaluation
# =============================================================================

def parse_response(response: str, expected_genes: List[str]) -> Dict[str, List[str]]:
    """Parse LLM response to extract predictions."""
    import re

    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        json_str = json_match.group(0) if json_match else "{}"

    try:
        parsed = json.loads(json_str)
        predictions = parsed.get("predictions", parsed)

        result = {"UP": [], "DOWN": [], "NOT_DE": []}
        for key in ["UP", "DOWN", "NOT_DE"]:
            if key in predictions and isinstance(predictions[key], list):
                result[key] = [str(g) for g in predictions[key]]

        # Missing genes default to NOT_DE
        all_pred = set(result["UP"]) | set(result["DOWN"]) | set(result["NOT_DE"])
        missing = set(expected_genes) - all_pred
        result["NOT_DE"].extend(list(missing))

        return result
    except:
        return {"UP": [], "DOWN": [], "NOT_DE": expected_genes}


def evaluate_sample(predictions: Dict, genes: List[str], labels: List[str]) -> Dict:
    """Evaluate a single sample."""
    pred_list, true_list = [], []

    for gene, true_label in zip(genes, labels):
        if gene in predictions.get('UP', []):
            pred_label = 'up'
        elif gene in predictions.get('DOWN', []):
            pred_label = 'down'
        else:
            pred_label = 'not_de'

        pred_list.append(pred_label)
        true_list.append(true_label)

    correct = sum(p == t for p, t in zip(pred_list, true_list))
    return {
        'accuracy': correct / len(pred_list) if pred_list else 0,
        'predictions': pred_list,
        'ground_truth': true_list
    }


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark(
    client: OpenRouterClient,
    model: str,
    test_samples: List[Dict],
    training_examples: List[Dict],
    n_shots: int,  # 0=zeroshot, None=all, N=random N
    save_path: Optional[str] = None
) -> Dict:
    """Run benchmark with specified n_shots."""

    is_zeroshot = (n_shots == 0)
    n_actual = len(training_examples) if n_shots is None else n_shots
    style = "zeroshot" if is_zeroshot else f"fewshot-{n_actual}"

    print(f"\n{'='*60}")
    print(f"Running: {style}")
    print(f"  Test samples: {len(test_samples)}")
    if not is_zeroshot:
        print(f"  Training examples: {n_actual}" + (" (all)" if n_shots is None else " (random)"))
    print(f"{'='*60}")

    all_preds, all_truths = [], []
    detailed_results = []
    system_prompt = SYSTEM_PROMPT_ZEROSHOT if is_zeroshot else SYSTEM_PROMPT_FEWSHOT

    for sample in tqdm(test_samples, desc=f"  {style}"):
        pert, genes, labels = sample['pert'], sample['genes'], sample['labels']

        # Create prompt
        if is_zeroshot:
            user_prompt = create_zeroshot_prompt(pert, genes)
        else:
            user_prompt = create_fewshot_prompt(pert, genes, training_examples, n_shots)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = client.chat(model=model, messages=messages)
            content = response["content"]
            predictions = parse_response(content, genes)
        except Exception as e:
            print(f"  Error: {e}")
            predictions = {"UP": [], "DOWN": [], "NOT_DE": genes}
            content = f"ERROR: {e}"

        eval_result = evaluate_sample(predictions, genes, labels)
        all_preds.extend(eval_result['predictions'])
        all_truths.extend(eval_result['ground_truth'])

        detailed_results.append({
            'pert': pert,
            'genes': genes,
            'ground_truth': labels,
            'predictions': predictions,
            'accuracy': eval_result['accuracy'],
            'prompt': user_prompt,
            'response': content[:2000]
        })

        time.sleep(0.3)

    # Compute metrics
    results = evaluate_three_class(all_preds, all_truths, labels=['down', 'not_de', 'up'])
    results['style'] = style
    results['n_shots'] = n_actual if not is_zeroshot else 0
    results['n_samples'] = len(test_samples)

    # Save
    if save_path:
        with open(save_path, 'w') as f:
            json.dump({
                'model': model,
                'style': style,
                'n_shots': results['n_shots'],
                'metrics': results,
                'samples': detailed_results
            }, f, indent=2)
        print(f"  Saved: {save_path}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM Benchmark: Compare prompting strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare 0 vs 10 vs all (default)
    python llm_benchmark_full.py --max_samples 5

    # Zero-shot only
    python llm_benchmark_full.py --n_shots 0 --max_samples 5

    # 10 random examples
    python llm_benchmark_full.py --n_shots 10 --max_samples 5

    # 20 random examples
    python llm_benchmark_full.py --n_shots 20 --max_samples 5

    # All training examples
    python llm_benchmark_full.py --n_shots all --max_samples 5
        """
    )
    parser.add_argument("--dataset", type=str, default="adamson")
    parser.add_argument("--n_shots", type=str, default="compare",
                        help="'compare', 'all', or number (0=zeroshot, 10, 20, etc)")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--genes_per_sample", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="../output/llm_benchmark_full")
    parser.add_argument("--model", type=str, default="claude-sonnet-4.5")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading {args.dataset}...")
    data = load_and_prepare_data(args.dataset, use_cache=True)

    print("\nClass distribution:")
    for split in ['train', 'test']:
        dist = data[split]['class_distribution']
        total = sum(dist.values())
        print(f"  {split}: UP={dist['up']/total*100:.1f}%, DOWN={dist['down']/total*100:.1f}%, NOT_DE={dist['not_de']/total*100:.1f}%")

    # Prepare data
    random.seed(42)
    test_samples = prepare_test_samples(data, args.max_samples, args.genes_per_sample)
    training_examples = prepare_training_examples(data)

    print(f"\nTest: {len(test_samples)} perturbations x {args.genes_per_sample} genes each")
    print(f"Train: {len(training_examples)} perturbations available")

    # Initialize client
    client = OpenRouterClient()

    # Determine runs
    if args.n_shots == "compare":
        runs = [(0, "zeroshot"), (10, "fewshot-10-random"), (None, f"fewshot-all-{len(training_examples)}")]
    elif args.n_shots == "all":
        runs = [(None, f"fewshot-all-{len(training_examples)}")]
    else:
        n = int(args.n_shots)
        name = "zeroshot" if n == 0 else f"fewshot-{n}-random"
        runs = [(n, name)]

    # Run benchmarks
    all_results = []
    for n_shots, name in runs:
        save_path = output_dir / f"{args.model}_{name}_{args.dataset}.json"
        results = run_benchmark(client, args.model, test_samples, training_examples, n_shots, str(save_path))
        all_results.append((name, results))

        print(f"\n  Results: Acc={results['accuracy']:.3f}, Macro-F1={results['macro_avg']['f1']:.3f}")

    # Summary table
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)

        # Header
        header = f"{'Metric':<12}"
        for name, _ in all_results:
            header += f"{name:>22}"
        print(header)
        print("-"*80)

        # Rows
        for metric in ['Accuracy', 'Macro F1', 'UP F1', 'DOWN F1', 'NOT_DE F1']:
            row = f"{metric:<12}"
            for _, r in all_results:
                if metric == 'Accuracy':
                    val = r['accuracy']
                elif metric == 'Macro F1':
                    val = r['macro_avg']['f1']
                else:
                    cls = metric.split()[0].lower()
                    val = r['per_class'].get(cls, {}).get('f1', 0)
                row += f"{val:>22.4f}"
            print(row)

        print("="*80)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
