import os
import re
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_results_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse metadata from first line
    meta_line = lines[0].strip()
    meta_match = re.search(r'pool_factor=(\d+), hidden_size=(\d+)\).*?(\d[\d,]*)', meta_line)
    metadata = {
        'pool_factor': int(meta_match.group(1)),
        'hidden_size': int(meta_match.group(2)),
        'trainable_params': int(meta_match.group(3).replace(',', ''))
    } if meta_match else {}

    val_loss = []
    val_ccc = []
    val_pcc = []

    for line in lines:
        line = line.strip()
        if 'Val Loss:' in line:
            loss_match = re.search(r'Val Loss: ([\d.]+)', line)
            ccc_match = re.search(r'Val CCC: ([\d.-]+)', line)
            pcc_match = re.search(r'Val PCC: ([\d.-]+)', line)

            if loss_match and ccc_match and pcc_match:
                val_loss.append(float(loss_match.group(1)))
                val_ccc.append(float(ccc_match.group(1)))
                val_pcc.append(float(pcc_match.group(1)))

    return {
        'meta': metadata,
        'val_loss': val_loss,
        'val_ccc': val_ccc,
        'val_pcc': val_pcc
    }

results_dir = "./"
files = glob(os.path.join(results_dir, "*.txt"))

all_results = {}

for f in files:
    name = os.path.basename(f)
    all_results[name] = parse_results_file(f)

def prepare_violin_data(metric_key):
    data = []
    for fname, result in all_results.items():
        values = result[metric_key]
        for v in values:
            data.append({'file': fname, metric_key: v})
    return pd.DataFrame(data)

# Create violin plots
# for metric in ['val_loss', 'val_ccc', 'val_pcc']:
#     df = prepare_violin_data(metric)
#     plt.figure(figsize=(10, 6))
#     sns.violinplot(x='file', y=metric, data=df)
#     plt.title(f'Violin Plot for {metric.upper()} (Last 10 Epochs)')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.show()

# convert the results to a DataFrame for easier manipulation
# print maximum/minimum/average/median/std of each metric the argmax argmin for which file put them 
def print_metrics_summary(metric_key):
    print(f"Summary for {metric_key.upper()}:")
    for fname, result in all_results.items():
        values = result[metric_key]
        if values:
            max_val = max(values)
            min_val = min(values)
            avg_val = sum(values) / len(values)
            median_val = sorted(values)[len(values) // 2]
            std_val = (sum((x - avg_val) ** 2 for x in values) / len(values)) ** 0.5
            print(f"Max: {max_val}, Min: {min_val}, Avg: {avg_val}, Median: {median_val}, Std: {std_val}",
                  f"Argmax: at {values.index(max_val)}",
                  f"Argmin: at {values.index(min_val)}", f"({fname})",
                  f"N-params: {result['meta']['trainable_params']}",)
    print("\n")
    print("====================================\n")
print_metrics_summary('val_loss')
print_metrics_summary('val_ccc')
print_metrics_summary('val_pcc')