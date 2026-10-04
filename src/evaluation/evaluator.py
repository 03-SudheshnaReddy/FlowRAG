"""
Evaluator for running evaluation on saved results.
"""

import os
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from loguru import logger

from src.evaluation.metrics import compute_qa_metrics, compute_forgetting
from src.models import create_generator


class Evaluator:
    """
    Evaluator for FlowRAG results.
    
    Loads saved prompts and generates answers using LLM,
    then computes evaluation metrics.
    
    Args:
        output_dir: Directory containing saved results
        model_name: Generator model name
        max_new_tokens: Maximum tokens to generate
    """
    
    METRICS = ['em', 'f1', 'bleu1', 'bleu2', 'bleu3', 'bleu4']
    
    def __init__(
        self,
        output_dir: str = "./output",
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        max_new_tokens: int = 64
    ):
        self.output_dir = output_dir
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        
        self._generator = None
    
    @property
    def generator(self):
        """Lazy load generator."""
        if self._generator is None:
            self._generator = create_generator(
                model_name=self.model_name,
                max_new_tokens=self.max_new_tokens,
                use_vllm=True
            )
            self._generator.load()
        return self._generator
    
    def evaluate_file(self, filepath: str) -> Dict[str, float]:
        """
        Evaluate a single results file.
        
        Args:
            filepath: Path to CSV file with prompts and answers
            
        Returns:
            Dictionary of average metrics
        """
        df = pd.read_csv(filepath)
        prompts = df['prompt'].tolist()
        answers = df['answer'].tolist()
        
        # Generate predictions
        logger.info(f"Generating predictions for {len(prompts)} samples...")
        predictions = self.generator.generate(prompts)
        
        # Compute metrics
        all_metrics = {m: [] for m in self.METRICS}
        
        for pred, ans in zip(predictions, answers):
            metrics = compute_qa_metrics(pred, ans)
            for m in self.METRICS:
                all_metrics[m].append(metrics[m])
        
        # Average metrics
        avg_metrics = {m: np.mean(scores) for m, scores in all_metrics.items()}
        
        # Save predictions
        pred_filepath = filepath.replace('.csv', '_predictions.csv')
        df['prediction'] = predictions
        df.to_csv(pred_filepath, index=False)
        logger.info(f"Saved predictions to: {pred_filepath}")
        
        return avg_metrics
    
    def evaluate_all(
        self,
        num_tasks: int,
        cl_method: str = 'fp'
    ) -> Dict[str, Any]:
        """
        Evaluate all task results.
        
        Args:
            num_tasks: Number of tasks
            cl_method: Continual learning method used
            
        Returns:
            Dictionary with metrics and forgetting scores
        """
        metric_matrix = {m: np.zeros((num_tasks, num_tasks)) for m in self.METRICS}
        file_metrics = {}  # Store per-file metrics
        
        for train_id in range(num_tasks):
            if cl_method == 'offline':
                # Offline: only diagonal
                eval_range = [train_id]
            else:
                # Online: lower triangle
                eval_range = range(train_id + 1)
            
            for eval_id in eval_range:
                filepath = os.path.join(self.output_dir, f"{train_id}_{eval_id}.csv")
                
                if not os.path.exists(filepath):
                    logger.warning(f"File not found: {filepath}")
                    continue
                
                metrics = self.evaluate_file(filepath)
                
                # Store per-file metrics
                file_key = f"{train_id}_{eval_id}"
                file_metrics[file_key] = {m: metrics[m] for m in self.METRICS}
                
                for m in self.METRICS:
                    metric_matrix[m][train_id, eval_id] = metrics[m]
        
        # Compute summary statistics
        results = {
            'metric_matrix': metric_matrix,
            'file_metrics': file_metrics,  # Add per-file metrics
            'average': {},
            'forgetting': {}
        }
        
        for m in self.METRICS:
            if cl_method == 'offline':
                # Average of diagonal
                results['average'][m] = np.diag(metric_matrix[m]).mean()
                results['forgetting'][m] = 0.0
            else:
                # Average of last row
                results['average'][m] = metric_matrix[m][-1, :].mean()
                results['forgetting'][m] = compute_forgetting(metric_matrix[m], m)
        
        return results
    
    def print_results(self, results: Dict[str, Any], method_name: str = "FlowRAG"):
        """Print formatted results."""
        print(f"\n{'='*50}")
        print(f"Results for: {method_name}")
        print(f"{'='*50}")
        
        print(f"\n{'Metric':<15} {'Average':>12} {'Forgetting':>12}")
        print("-" * 40)
        
        for m in self.METRICS:
            avg = results['average'][m] * 100
            fgt = results['forgetting'][m] * 100
            print(f"{m.upper():<15} {avg:>11.2f}% {fgt:>11.2f}%")
        
        print(f"{'='*50}\n")


def main():
    """Command-line interface for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate FlowRAG results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Base directory for outputs')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name subfolder (e.g., flowrag_nq_covidqa)')
    parser.add_argument('--num_tasks', type=int, default=4,
                        help='Number of tasks (datasets)')
    parser.add_argument('--cl_method', type=str, default='fp',
                        choices=['offline', 'fp', 'replug', 'emdr', 'fid', 'l2r'],
                        help='Continual learning method used during training')
    parser.add_argument('--generator', type=str, 
                        default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Generator model for answer generation')
    parser.add_argument('--max_new_tokens', type=int, default=64,
                        help='Maximum tokens to generate')
    
    args = parser.parse_args()
    
    # Build full output directory path
    output_dir = os.path.join(args.output_dir, args.exp_name) if args.exp_name else args.output_dir
    
    logger.info(f"Evaluating results from: {output_dir}")
    logger.info(f"Number of tasks: {args.num_tasks}")
    logger.info(f"CL Method: {args.cl_method}")
    
    evaluator = Evaluator(
        output_dir=output_dir,
        model_name=args.generator,
        max_new_tokens=args.max_new_tokens
    )
    
    results = evaluator.evaluate_all(
        num_tasks=args.num_tasks,
        cl_method=args.cl_method
    )
    
    evaluator.print_results(results, method_name=args.cl_method.upper())
    
    # Save results to JSON
    import json
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        'average': results['average'],
        'forgetting': results['forgetting'],
        'file_metrics': results['file_metrics'],  # Per-file metrics including F1
        'metric_matrix': {m: results['metric_matrix'][m].tolist() for m in evaluator.METRICS}
    }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Save results to TXT table format
    txt_file = os.path.join(output_dir, 'evaluation_results.txt')
    with open(txt_file, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"FlowRAG Evaluation Results - {args.cl_method.upper()}\n")
        f.write(f"{'='*60}\n\n")
        
        # Summary table
        f.write("Summary Metrics:\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'Metric':<15} {'Average':>12} {'Forgetting':>12}\n")
        f.write("-" * 45 + "\n")
        for m in evaluator.METRICS:
            avg = results['average'][m] * 100
            fgt = results['forgetting'][m] * 100
            f.write(f"{m.upper():<15} {avg:>11.2f}% {fgt:>11.2f}%\n")
        f.write("-" * 45 + "\n\n")
        
        # Metric matrices
        for m in evaluator.METRICS:
            f.write(f"\n{m.upper()} Matrix (Train x Eval):\n")
            f.write("-" * 50 + "\n")
            matrix = results['metric_matrix'][m]
            # Header
            header = "Train\\Eval"
            f.write(f"{header:<10}")
            for j in range(args.num_tasks):
                f.write(f"Task{j:<6}")
            f.write("\n")
            f.write("-" * 50 + "\n")
            # Data rows
            for i in range(args.num_tasks):
                f.write(f"Task{i:<6}")
                for j in range(args.num_tasks):
                    val = matrix[i, j] * 100
                    f.write(f"{val:>9.2f}%")
                f.write("\n")
            f.write("\n")
        
        f.write(f"{'='*60}\n")
    
    logger.info(f"TXT table saved to: {txt_file}")


if __name__ == "__main__":
    main()
