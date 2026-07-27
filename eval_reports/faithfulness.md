# Faithfulness & Citation Evaluation

Full-method pipeline over **7** evaluation questions (single method; no baseline).
Generation model: **mistral-large-latest**. Faithfulness = fraction of atomic claims entailed by their cited source span.

## Headline (aggregate)

| Metric | Value |
|---|---|
| Claim faithfulness (NLI) | 0.4220 |
| Claim faithfulness (LLM judge) | 0.7765 |
| NLI–judge agreement | 0.5332 |
| Citation precision (vs CUAD gold spans) | 0.1815 |
| Citation recall (vs CUAD gold spans) | 1.0000 |
| Citation IoU (char offsets) | 0.1815 |

## Per-question

| Question (contract :: clause) | Claims | Faith (NLI) | Faith (Judge) | Agree | Cit P | Cit R | Cit IoU |
|---|---|---|---|---|---|---|---|
| ACME_SUPPLY_AGREEMENT::Cap On Liability | 7 | 0.5714 | 0.7143 | 0.8571 | 0.2640 | 1.0000 | 0.2640 |
| ACME_SUPPLY_AGREEMENT::Governing Law | 4 | 0.2500 | 1.0000 | 0.2500 | 0.0825 | 1.0000 | 0.0825 |
| ACME_SUPPLY_AGREEMENT::Renewal Term | 9 | 0.7778 | 0.6667 | 0.8889 | 0.1425 | 1.0000 | 0.1425 |
| ACME_SUPPLY_AGREEMENT::Termination For Convenience | 4 | 0.0000 | 1.0000 | 0.0000 | 0.3000 | 1.0000 | 0.3000 |
| GLOBEX_LICENSE::Anti-Assignment | 11 | 0.4545 | 0.4545 | 0.6364 | 0.2300 | 1.0000 | 0.2300 |
| GLOBEX_LICENSE::Governing Law | 5 | 0.2000 | 0.8000 | 0.4000 | 0.1500 | 1.0000 | 0.1500 |
| GLOBEX_LICENSE::License Grant | 10 | 0.7000 | 0.8000 | 0.7000 | 0.1017 | 1.0000 | 0.1017 |
