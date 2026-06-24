import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

def main():
    results_dir = os.path.join(_PROJECT_ROOT, 'data', 'eval_results')
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory {results_dir} does not exist.")
        sys.exit(1)

    # Read all JSON files
    data_by_mode = {}
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and filename.startswith('results_'):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    res = json.load(f)
                mode = res['mode']
                severity = res['severity']
                success_rate = res['success_rate']
                cube_miss_rate = res.get('cube_miss_rate', 0.0)
                target_miss_rate = res.get('target_miss_rate', 0.0)
                overall_miss_rate = res.get('overall_miss_rate', 0.0)

                if mode not in data_by_mode:
                    data_by_mode[mode] = {}
                
                data_by_mode[mode][severity] = {
                    'success_rate': success_rate,
                    'cube_miss_rate': cube_miss_rate,
                    'target_miss_rate': target_miss_rate,
                    'overall_miss_rate': overall_miss_rate
                }
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if not data_by_mode:
        print("No evaluation data found to plot.")
        sys.exit(1)

    print("Found evaluation data for modes:", list(data_by_mode.keys()))

    severities = [0, 1, 2, 3]
    severity_labels = ['0 (Clean)', '1 (Low)', '2 (Medium)', '3 (High)']

    # Plot 1: Success Rate vs Corruption Severity
    plt.figure(figsize=(8, 6))
    
    # Modern styling
    plt.style.use('seaborn-v0_8-whitegrid')
    
    colors = {
        'baseline': '#E24A33',    # Red-Orange
        'yolo_guided': '#348ABD',  # Blue
        'yolo_crops': '#988ED5',   # Purple
        'gt_boxes': '#8EBA42'      # Green
    }
    
    markers = {
        'baseline': 'o',
        'yolo_guided': 's',
        'yolo_crops': '^',
        'gt_boxes': 'D'
    }

    for mode, mode_data in data_by_mode.items():
        y_vals = []
        x_vals = []
        for sev in severities:
            if sev in mode_data:
                x_vals.append(sev)
                y_vals.append(mode_data[sev]['success_rate'] * 100) # percentage
        
        # Sort by x_vals
        sort_idx = np.argsort(x_vals)
        x_vals = np.array(x_vals)[sort_idx]
        y_vals = np.array(y_vals)[sort_idx]

        color = colors.get(mode, None)
        marker = markers.get(mode, 'o')
        
        plt.plot(x_vals, y_vals, label=mode, marker=marker, linewidth=2, markersize=8, color=color)

    plt.title("Robustness Curve: Success Rate vs. Corruption Severity", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Corruption Severity", fontsize=12)
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.xticks(severities, severity_labels)
    plt.ylim(-5, 105)
    plt.legend(frameon=True, fontsize=11, loc='lower left')
    plt.tight_layout()
    
    save_path1 = os.path.join(_PROJECT_ROOT, 'data', 'robustness_curve.png')
    plt.savefig(save_path1, dpi=300)
    print(f"Saved success rate plot to {save_path1}")
    plt.close()

    # Plot 2: Detection Miss Rate vs Corruption Severity (for YOLO-based modes)
    plt.figure(figsize=(8, 6))
    
    has_miss_data = False
    for mode, mode_data in data_by_mode.items():
        if mode not in ['yolo_guided', 'yolo_crops']:
            continue
        
        y_vals = []
        x_vals = []
        for sev in severities:
            if sev in mode_data:
                x_vals.append(sev)
                y_vals.append(mode_data[sev]['overall_miss_rate'] * 100)
                has_miss_data = True
                
        # Sort by x_vals
        if x_vals:
            sort_idx = np.argsort(x_vals)
            x_vals = np.array(x_vals)[sort_idx]
            y_vals = np.array(y_vals)[sort_idx]
            
            color = colors.get(mode, None)
            marker = markers.get(mode, 'o')
            plt.plot(x_vals, y_vals, label=f"{mode} (overall miss)", marker=marker, linewidth=2, markersize=8, color=color)

    if has_miss_data:
        plt.title("YOLO Detection Miss Rate vs. Corruption Severity", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Corruption Severity", fontsize=12)
        plt.ylabel("Detection Miss Rate (%)", fontsize=12)
        plt.xticks(severities, severity_labels)
        plt.ylim(-5, 105)
        plt.legend(frameon=True, fontsize=11, loc='upper left')
        plt.tight_layout()
        
        save_path2 = os.path.join(_PROJECT_ROOT, 'data', 'detection_miss_rate.png')
        plt.savefig(save_path2, dpi=300)
        print(f"Saved detection miss rate plot to {save_path2}")
    plt.close()

if __name__ == '__main__':
    main()
