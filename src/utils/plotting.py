import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os

TASKS = {'upper_color': 11, "lower_color": 11, "gender": 1, "bag": 1, "hat": 1}
METRICS = ["loss", "accuracy", "precision", "recall", "f1"]

def plot_results(results_dir: Path | str = "runs/") -> None:
    """Plot training, validation, and test metrics.
    
    Parameters
    ----------
    results_dir : Path | str
        The directory containing the training and validation results.
    """
    # Convert to Path object if needed
    results_dir = Path(results_dir) if not isinstance(results_dir, Path) else results_dir
    
    # Create plots directory if it doesn't exist
    plots_dir = results_dir / "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Check which data splits are available
    splits = []
    if (results_dir / "train").exists():
        splits.append("train")
    if (results_dir / "val").exists():
        splits.append("val")
    if (results_dir / "test").exists():
        splits.append("test")
    
    if not splits:
        print("No data found. Please check the results directory.")
        return
        
    # Process each task
    for task in TASKS:
        print(f"Plotting results for task: {task}")
        
        # Load data for each available split
        data = {}
        for split in splits:
            file_path = results_dir / split / f"metrics_task_{task}.csv"
            if file_path.exists():
                # Read the CSV and clean column names by stripping whitespace
                df = pd.read_csv(file_path)
                df.columns = [col.strip() for col in df.columns]
                data[split] = df
                
                print(f"{split.capitalize()} data columns (after cleaning): {data[split].columns.tolist()}")
                
                # Add epoch column if not present (using index)
                if "epoch" not in data[split].columns:
                    data[split]["epoch"] = data[split].index
                
                print(f"{split.capitalize()} data loaded, shape: {data[split].shape}")
            else:
                print(f"Warning: No {split} data found at {file_path}")
                
        if not data:
            print(f"No data found for task {task}, skipping.")
            continue
        
        # Map from our metric names to actual column names (which might differ slightly)
        metric_mapping = {}
        for metric in METRICS:
            for column in data[list(data.keys())[0]].columns:
                if metric in column.lower():
                    metric_mapping[metric] = column
                    break
        
        print(f"Metric mapping for task {task}: {metric_mapping}")
                
        # Create one figure per metric
        for metric, column_name in metric_mapping.items():
            # Check if the metric exists in all dataframes
            if not all(column_name in df.columns for df in data.values()):
                print(f"Skipping {metric} for task {task} as it's not available in all splits")
                continue
                
            fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
            
            for split, df in data.items():
                sns.lineplot(data=df, x="epoch", y=column_name, label=f"{split.capitalize()}", ax=ax)
            
            ax.set_title(f"{task} - {metric.capitalize()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(plots_dir / f"{task}_{metric}.png", dpi=300)
            plt.close(fig)
        
        # Create a summary figure with all metrics in subplots
        if data:
            available_metrics = [m for m, col in metric_mapping.items() 
                               if all(col in df.columns for df in data.values())]
            
            if available_metrics:
                n_metrics = len(available_metrics)
                fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 5 * n_metrics), sharex=True)
                if n_metrics == 1:
                    axes = [axes]  # Make axes indexable if only one subplot
                
                for i, metric in enumerate(available_metrics):
                    column_name = metric_mapping[metric]
                    for split, df in data.items():
                        sns.lineplot(data=df, x="epoch", y=column_name, label=f"{split.capitalize()}", ax=axes[i])
                    
                    axes[i].set_title(f"{metric.capitalize()}")
                    axes[i].set_ylabel(metric.capitalize())
                    axes[i].grid(True, alpha=0.3)
                    axes[i].legend()
                
                axes[-1].set_xlabel("Epoch")
                fig.suptitle(f"Task: {task}", fontsize=16)
                plt.tight_layout()
                plt.subplots_adjust(top=0.95)
                plt.savefig(plots_dir / f"{task}_summary.png", dpi=300)
                plt.close(fig)
            
if __name__ == "__main__":
    # Example usage
    plot_results("runs/")