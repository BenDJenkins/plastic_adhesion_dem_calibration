# 🔬 pyspc: Nanoindentation & EEPA Calibration Pipeline

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![SciPy](https://img.shields.io/badge/SciPy-Optimization-orange.svg)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-success.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-IEEE_Style-yellow.svg)

**`pyspc`** is an automated Python pipeline for processing experimental nanoindentation data and calibrating the **Elastic-Plastic Adhesion** models.

It handles everything from raw `.nsa` file parsing and artifact removal to differential evolution optimization and publication-ready plotting.

## 🚀 Quick Start

### 1. Build the Inventory
Review your data before running heavy optimizations. `pyspc` generates a clean, tree-structured terminal report of your `.nsa` files and particle radii.

```python
import pyspc

target_dir = './raw_data/Aggregated'
base_dir = './raw_data'

# Scans directories, extracts diameters, and prompts for user confirmation
df_inventory = pyspc.build_preprocessing_registry(target_dir, base_dir)
```

### 2. Batch Optimization

Run the EEPA physics-based calibration across all approved files.

```python
# Runs SciPy differential evolution on all valid datasets
df_results = pyspc.process_and_save_data(df_inventory, output_csv="optimized_results.csv")
```

### 3. Generate Visualizations

Create an interactive HTML summary or export an IEEE-style graph.

```python
import pandas as pd

df = pd.read_csv("optimized_results.csv")

# Generate the interactive hover-plot dashboard
pyspc.create_interactive_plot(df, output_html='interactive_summary.html')

# Generate a static, publication-ready vector graphic
pyspc.create_publication_plot(df, output_img='eepa_summary.pdf')
```