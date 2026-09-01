# Association Rule Mining — CMPE 255 Assignment 1 Part 2

This directory contains a reproduction-ready market-basket analysis inspired by the popular Kaggle **Groceries** transaction dataset. The experiment follows CRISP-DM from business understanding through evaluation and deployment-oriented recommendations.

> ## Dataset and Execution Disclaimer
>
> The original transaction dataset is **not included and was not executed** because of dataset-access constraints. The complete methodology and code are reproduction-ready, but actual frequent itemsets, association rules, support, confidence, lift, and data-dependent business findings are intentionally not reported. No numerical results have been fabricated.

## Files

- `association_rule_mining.ipynb` — documented CRISP-DM experiment and executable analysis pipeline.
- `requirements.txt` — pinned-minimum Python dependencies.
- `prompt.txt` — the exact assignment prompt used to create this experiment.
- `images/` — destination for figures generated when the notebook runs with real data.

## Dataset setup

1. Download a Groceries-style CSV from its authorized source. Do not commit it if its license or course rules prohibit redistribution.
2. Either place it at `data/Groceries_dataset.csv` (relative to this directory), or set `ARM_DATASET_PATH` to its location:

   ```bash
   export ARM_DATASET_PATH=/absolute/path/to/Groceries_dataset.csv
   ```

3. Expected columns are `Member_number`, `Date`, and `itemDescription`. Column matching is case-insensitive and also accepts common aliases documented in the notebook. A transaction is identified by the combination of member and date, which avoids combining all purchases by one member across different shopping dates.

## Reproduce the experiment

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab association_rule_mining.ipynb
```

Run **Kernel → Restart Kernel and Run All Cells**. Without a dataset, the notebook exits analytical stages safely, prints setup guidance, and produces no invented results. With a valid dataset, it validates the schema, reports quality diagnostics, cleans transactions, encodes baskets, mines frequent itemsets with Apriori, generates and filters rules, ranks useful rules, and writes plots to `images/`.

## Configuration

The notebook exposes reproducibility parameters near the top:

- `MIN_SUPPORT` — minimum itemset support.
- `MIN_CONFIDENCE` — minimum rule confidence.
- `MIN_LIFT` — minimum lift for useful-rule filtering.
- `TOP_N_RULES` — number of ranked rules displayed/visualized.

Thresholds must be selected in context after inspecting the real dataset. Changing them changes the discovered patterns; report the final settings alongside any subsequently executed results.

## Responsible interpretation

Association indicates co-occurrence, not causation. Promotions, seasonality, store layout, stock availability, repeated customers, and data-collection choices can confound rules. Validate candidate actions with holdout periods and controlled experiments before deployment. Avoid using customer identifiers for profiling; they are used only to construct transaction keys.
