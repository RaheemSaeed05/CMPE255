# KDD — Bakery Market Basket Analysis

Discover historical bakery items purchased together using Apriori association rules, then use the retained rules as candidates for single-item basket suggestions. From **9,465 baskets and 94 distinct items**, Apriori produced **61 frequent itemsets and 14 retained rules**. The leading rule, **Toast → Coffee**, has **2.37% support, 70.44% confidence, and 1.4724 lift**. All 14 rules passed direct basket-count verification. These findings describe historical associations; they do not demonstrate future recommendation accuracy, causation, or increased sales.

## Files and project structure

| File | Purpose | Status in this documentation delivery |
|---|---|---|
| `README.md` | Project overview, reproduction instructions, and delivery checklist | Included |
| `report.md` | Five KDD phases, checks, results, limitations, and status | Included |
| `03_KDD_Bakery_Basket_Analysis.ipynb` | Executed notebook with the five KDD phases and their outputs | Referenced project notebook; add alongside these files |
| `Bakery.csv` | Original input dataset from Kaggle | Obtain separately; not bundled |
| Saved-rule / suggestion artifacts | Reusable rules and basket-suggestion demonstration | Pending in the approved Phase 5 record; names and exports not verified |

The table describes the intended project layout, not a claim that the notebook, data, or saved-rule artifacts are included in this two-file documentation package.

## Reproduce the five KDD phases in Colab

1. Download the [Bakery dataset on Kaggle](https://www.kaggle.com/datasets/akashdeepkuila/bakery) and obtain `Bakery.csv`. The dataset used in the recorded execution has **20,507 item rows and five columns**: `TransactionNo`, `Items`, `DateTime`, `Daypart`, and `DayType`.
2. Open `03_KDD_Bakery_Basket_Analysis.ipynb` in Google Colab. The selection code prompts for a `Bakery.csv` upload; use the exact expected filename. Run the cells in order so `selected_df`, `clean_df`, `basket_matrix`, `rules`, and the evaluation outputs are created sequentially.
3. Use a Python environment with **pandas**, **NumPy**, and **mlxtend**. The recorded Apriori execution used **mlxtend 0.23.4**. A clean-environment reinstall has not been verified here. If the library is unavailable in Colab, install the recorded version:

   ```python
   %pip install mlxtend==0.23.4
   ```

4. Inspect the five phases and their assertions. Selection preserves all original rows; Preprocessing validates values and timestamps, trims labels, and collapses repeated transaction-item pairs; Transformation creates a Boolean basket matrix; Data Mining runs Apriori; Interpretation and Evaluation recomputes metrics from basket counts.
5. Preserve the executed outputs and the two expert-review rounds for each phase in your final notebook or submission records. The reviewer approvals are recorded in the project conversation; their inclusion in any particular notebook file has not been independently checked during this documentation preparation.

## Fixed mining settings and output

| Setting | Value |
|---|---:|
| Historical baskets (support denominator) | 9,465 |
| Minimum support | 0.01 (at least 95 baskets) |
| Minimum confidence | 0.30 |
| Required lift | Greater than 1 |
| Maximum frequent-itemset size | 3 |
| Consequent restriction | Exactly one item |
| Rule ordering | Lift, confidence, support (descending), then item names (ascending) |

The Boolean matrix contains **9,465 transaction rows × 94 item columns** and **18,887 True values**. All transactions are included, even the **3,948 single-item baskets**, because they belong in the denominator for support. Rules use presence or absence, not purchase quantities, customer sequences, or spending.

## Rule verification and interpretation

All **14 rules** had support, confidence, and lift recalculated directly from the cleaned baskets. The results agreed with `mlxtend` within `rtol=1e-9` and `atol=1e-12`.

For **Toast → Coffee**, **318** baskets contained Toast, **4,528** contained Coffee, and **224** contained both, out of **9,465** baskets. This yields **2.3666% support**, **70.4403% confidence**, **47.8394% overall Coffee frequency**, and **1.4724 lift**. Confidence exceeds the overall Coffee baseline by **22.60 percentage points**, but the rule applies to only **2.37%** of all baskets. **Thirteen** retained rules have Coffee as their consequent; the remaining rule is **Pastry → Bread**, with lift around **1.0350**. These limitations matter when interpreting suggestion variety and usefulness.
