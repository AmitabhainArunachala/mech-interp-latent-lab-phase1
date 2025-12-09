"""
2D Subsystem Map Visualization

Generates scatter plot of R_V vs Attention Entropy with subsystem coloring.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


def load_all_results(results_dir):
    """
    Load all result JSON files from results directory.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        list: List of result dictionaries
    """
    results_dir = Path(results_dir)
    all_results = []
    
    # Known subsystems from validation
    known_results = {
        "retrieval": {"r_v": 1.41, "attention_entropy": 0.12},
        "meta_cognitive": {"r_v": 0.58, "attention_entropy": 0.23},
        "logic": {"r_v": 0.60, "attention_entropy": 0.23}
    }
    
    # Add known results
    for subsystem, coords in known_results.items():
        all_results.append({
            "subsystem": subsystem,
            "r_v": coords["r_v"],
            "attention_entropy": coords["attention_entropy"],
            "source": "validation"
        })
    
    # Load new results from JSON files
    for json_file in results_dir.glob("**/*_results.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            for result in data:
                if 'r_v' in result and 'attention_entropy' in result:
                    result['source'] = 'measured'
                    all_results.append(result)
    
    return all_results


def create_2d_map(results, output_path=None):
    """
    Create 2D scatter plot of R_V vs Attention Entropy.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save figure
    """
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Group by subsystem for averaging
    subsystem_data = df.groupby('subsystem').agg({
        'r_v': 'mean',
        'attention_entropy': 'mean'
    }).reset_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color map for subsystems
    subsystems = subsystem_data['subsystem'].unique()
    colors = sns.color_palette("husl", len(subsystems))
    color_map = dict(zip(subsystems, colors))
    
    # Plot each subsystem
    for subsystem in subsystems:
        data = subsystem_data[subsystem_data['subsystem'] == subsystem]
        color = color_map[subsystem]
        
        ax.scatter(
            data['r_v'],
            data['attention_entropy'],
            label=subsystem,
            color=color,
            s=200,
            alpha=0.7,
            edgecolors='black',
            linewidths=2
        )
    
    # Add individual points if available
    if len(df) > len(subsystem_data):
        for subsystem in df['subsystem'].unique():
            subset = df[df['subsystem'] == subsystem]
            color = color_map.get(subsystem, 'gray')
            ax.scatter(
                subset['r_v'],
                subset['attention_entropy'],
                color=color,
                alpha=0.3,
                s=50
            )
    
    # Labels and title
    ax.set_xlabel('R_V (Participation Ratio)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Attention Entropy (Normalized)', fontsize=14, fontweight='bold')
    ax.set_title('2D Subsystem Map: R_V vs Attention Entropy', 
                 fontsize=16, fontweight='bold')
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Add reference lines
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='R_V = 1.0')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {output_path}")
    
    return fig


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate 2D subsystem map")
    parser.add_argument("--results-dir", type=str, 
                       default="03_RESULTS",
                       help="Directory containing result JSON files")
    parser.add_argument("--output", type=str,
                       default="04_FIGURES/complete_2d_map.png",
                       help="Output path for figure")
    
    args = parser.parse_args()
    
    # Load results
    print("Loading results...")
    results = load_all_results(args.results_dir)
    print(f"Loaded {len(results)} results")
    
    # Create visualization
    print("Generating 2D map...")
    fig = create_2d_map(results, output_path=args.output)
    
    plt.show()


if __name__ == "__main__":
    main()

