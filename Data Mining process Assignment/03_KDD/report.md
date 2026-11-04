# Discovering Bakery Purchase Associations with KDD

## Summary

This project applies the five-phase **Knowledge Discovery in Databases (KDD)** process to a historical bakery transaction dataset. Its objective is to find items frequently bought together and identify association-rule candidates for suggesting one additional item to a customer with an existing basket. The analysis uses **9,465 historical baskets** and **94 distinct item names**. Under predefined Apriori restrictions, it produced **61 frequent itemsets and 14 retained rules**, all of which passed direct count-based support, confidence, and lift verification.

The leading rule, **Toast → Coffee**, occurred jointly in **224 of 9,465 baskets** (2.3666% support). Among the **318** baskets containing Toast, **224** also contained Coffee (70.4403% confidence), compared with Coffee's **47.8394%** overall basket frequency. Its lift is **1.4724**. These are descriptive historical associations, not evidence that a recommendation causes purchases or that rules predict future transactions accurately.

## 1. Selection

The data source is the [Bakery dataset on Kaggle](https://www.kaggle.com/datasets/akashdeepkuila/bakery), using `Bakery.csv`. The supplied file had **20,507 item-level rows** and five fields: `TransactionNo`, `Items`, `DateTime`, `Daypart`, and `DayType`. One row represents an item entry associated with a transaction number, not an independently verified entire purchase event. The data contained **9,465 distinct transaction numbers** and **94 distinct item names**.

The analysis selected `TransactionNo` to identify baskets, `Items` to identify item membership, and `DateTime` for transaction-consistency checks. `Daypart` and `DayType` were excluded because the project targets overall co-purchase patterns rather than time-dependent patterns. All original item rows and their order were preserved during selection; the dataset was manageable in Colab. No supervised target variable was required.

Initial checks found **zero pandas-recognized missing values**, **zero blank text values**, and **1,620 repeated full rows**. The repeated rows were retained for preprocessing because they could reflect quantities or duplicate recordings; their meaning was not established. Treating each transaction number as one purchase event remained an assumption to be checked for internal timestamp consistency.

The intended result is an interpretable collection of historical co-purchase rules and a subsequent basket-suggestion demonstration, not a verified operational recommendation system.

## 2. Preprocessing

Preprocessing checked the selected fields for missing or blank values. `TransactionNo` was converted to a finite, positive integer; invalid values would stop execution. `DateTime` was parsed with the explicit format `%Y-%m-%d %H:%M:%S`, with parsing errors set to raise rather than silently coerce data. Each transaction number was checked for timestamp consistency; **zero transactions had inconsistent timestamps**.

Leading and trailing whitespace was removed from item labels, changing **21 item entries**. Capitalization and internal spelling were preserved. For presence-based association rules, duplicate `(TransactionNo, Items)` combinations were then collapsed, removing **1,620 repeated transaction-item rows**. This was a representation decision, not evidence that the original entries were erroneous or that their quantities were known.

The resulting cleaned dataset contained **18,887 unique transaction-item rows**, all **9,465 transaction numbers**, and **94 distinct items**. Checks confirmed unique transaction-item pairs, nonempty item names, no remaining missing values, and preservation of the transaction-number set. The parsed timestamps ranged from **2016-01-11 07:51:20** to **2017-12-03 16:28:00**.

A single timestamp per transaction supports internally consistent grouping within the file; it does **not** independently verify purchase events or the correctness of the historical dates. The dates were not used as mining features.

## 3. Transformation

The cleaned data was grouped by `TransactionNo`, yielding baskets of distinct item names. A Boolean transaction-by-item matrix was constructed with **9,465 rows × 94 columns**. `True` indicates an item is present in a transaction; `False` indicates it is absent. The matrix contains neither quantities nor purchasing order.

All **9,465 transactions** were retained, including **3,948 single-item baskets**, so the support denominator covers the full historical transaction set. Basket sizes ranged from **1 to 10 distinct items**, with a mean of approximately **2.00**.

Checks verified unique transaction and item labels, Boolean column types, no missing matrix values, at least one item per basket, and preservation of every transaction and item. The **18,887 True values** matched the **18,887 unique cleaned transaction-item pairs**. Finally, the items reconstructed from each matrix row matched the corresponding cleaned basket exactly. This establishes correctness of the representation relative to the cleaned records, not external truth about the purchases.

## 4. Data Mining

The notebook applied **Apriori** through **mlxtend 0.23.4** to discover frequent itemsets, followed by association-rule generation. Parameters were fixed before mining, rather than tuned against the reported results:

| Restriction | Fixed choice | Reason |
|---|---|---|
| Minimum support | **0.01** | Requires at least **95 of 9,465 baskets** to support an itemset |
| Minimum confidence | **0.30** | Consequent must appear in at least 30% of antecedent-containing baskets |
| Lift | **> 1** | Requires observed co-occurrence above the independence-based expectation |
| Maximum itemset size | **3** | Keeps patterns compact and interpretable |
| Consequents | **1 item** | Supports suggesting a single additional item |

Rules were ordered by lift, confidence, and support descending, followed by alphabetical antecedent and consequent labels to resolve ties. The execution reported **61 frequent itemsets** and **14 retained rules**. The code verified the thresholds, finite support/confidence/lift values, nonempty and disjoint rule sides, one-item consequents, and the maximum itemset size. All **9,465** baskets remained in the support denominator. These settings were project choices, not optimized thresholds.

The leading rules in the recorded ranking were:

| Antecedent → Consequent | Support | Confidence | Lift |
|---|---:|---:|---:|
| Toast → Coffee | 0.0237 | 0.7044 | 1.4724 |
| Spanish Brunch → Coffee | 0.0109 | 0.5988 | 1.2518 |
| Medialuna → Coffee | 0.0352 | 0.5692 | 1.1899 |
| Pastry → Coffee | 0.0475 | 0.5521 | 1.1542 |
| Alfajores → Coffee | 0.0197 | 0.5407 | 1.1302 |

The figures above are rounded display values. The rule list describes co-occurrence in the mining sample; it does not establish causality, purchasing order, future recommendation accuracy, or sales improvements.

## 5. Interpretation and Evaluation

All **14 retained rules** were evaluated using direct membership counts in baskets reconstructed from the cleaned data. For a rule `A → B`, the code calculated:

- **Support:** baskets containing both A and B divided by all 9,465 baskets.
- **Confidence:** baskets containing both A and B divided by baskets containing A.
- **Consequent baseline:** baskets containing B divided by all baskets.
- **Lift:** confidence divided by the consequent baseline.

The recalculated support, confidence, and lift matched the mining output for **all 14 rules** using `rtol=1e-9` and `atol=1e-12`. This confirms the reported metrics against direct basket counts; it is not a held-out predictive evaluation.

### Leading rule: Toast → Coffee

| Measure | Direct count or calculation | Result |
|---|---|---:|
| Total baskets | — | 9,465 |
| Toast baskets | — | 318 |
| Coffee baskets | — | 4,528 |
| Toast-and-Coffee baskets | — | 224 |
| Support | 224 / 9,465 | **2.3666%** |
| Confidence | 224 / 318 | **70.4403%** |
| Coffee baseline | 4,528 / 9,465 | **47.8394%** |
| Lift | (224 / 318) / (4,528 / 9,465) | **1.4724** |

Coffee appeared in Toast-containing baskets approximately **22.60 percentage points** more often than its overall historical frequency. That is a meaningful descriptive difference within this dataset, but the joint occurrence covers only **2.37%** of all baskets. The rule therefore has relatively high conditional confidence but limited overall coverage. Neither difference demonstrates the effect of actively recommending Coffee.

### Other retained rules

**Thirteen of the 14 rules** have Coffee as their consequent; the remaining rule is **Pastry → Bread**. This concentration restricts the variety of candidate suggestions. Pastry → Bread has **33.87% confidence** against Bread's **32.72%** overall frequency, giving only **1.0350 lift**. It satisfies the preselected restrictions but adds relatively little association above the baseline. Crossing fixed thresholds by itself does not establish practical value or statistical significance.

### Evaluation conclusion and limitations

The **historical discovery objective is supported**: the procedure generated qualifying association rules, and all reported rule metrics were verified against cleaned-basket counts. Evaluation reused the same baskets as mining. No independent future-period or held-out prediction test, customer-response study, statistical significance test, or causal measurement was performed. Accordingly, **future accuracy, customer behavior changes, and sales benefits remain unverified**.

Other limitations include dependence on the historical sample, fixed threshold choices, incomplete suggestion diversity, the presence-only treatment of repeated entries, and the lack of independent verification for transaction meaning and timestamps. Customer identity, units purchased, sequence, and spending were not established.

## Saved-rule suggestion demonstration

The approved Phase 5 record identifies this as **pending**. The planned next step is to save the mined rules and show an example basket receiving a candidate item not already present. A recommendation demonstration, exported rule artifact, or independent reproduction of a saved-rule workflow has **not been verified in the materials provided for this report**. No recommendation performance or operational deployment claims are made.

## Critique and revision evidence

All **five phases** underwent two expert-review rounds in the project conversation. Both rounds approved each phase with **no essential corrections**. The Revision 1 records explicitly retained the original code, settings, results, and limitations and did not claim additional execution. These approvals evaluate the supplied drafts, code, and outputs; they do not constitute independent external validation of the raw dataset.

## Artifacts, reproducibility, and submission status

The accompanying `README.md` describes the expected notebook, raw-data source, dependencies, rerun steps, and required delivery items. **mlxtend 0.23.4** was recorded during mining; complete fresh-environment reproducibility was not tested as part of writing these documents. The numbers in this report come from the user-provided executed code, outputs, and review record. Preparing the report did **not** rerun Apriori, re-open the raw CSV, or execute the saved-rule demonstration.

**Completed in the supplied record:** Selection, Preprocessing, Transformation, Data Mining, Interpretation and Evaluation, direct verification of 14 rules, and two review rounds for each phase. **Pending or not verified in that record:** saved-rule suggestion demonstration, final artifact packaging, Medium article, YouTube walkthrough, GitHub publication, and final submission links. This README and report provide the written documentation for the KDD project.

## Reference

- Dataset: [Bakery — Kaggle](https://www.kaggle.com/datasets/akashdeepkuila/bakery)

The dataset link identifies the source. All numerical findings above are drawn from the provided execution results; no additional external source verification was performed while preparing this report.
