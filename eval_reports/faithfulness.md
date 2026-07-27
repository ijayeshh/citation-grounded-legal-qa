# Faithfulness & Citation Evaluation

Full-method pipeline over **7** evaluation questions (single method; no baseline).
Faithfulness = fraction of atomic claims entailed by their cited source span.

## Headline (aggregate)

| Metric | Value |
|---|---|
| Claim faithfulness (NLI) | 0.4714 |
| Claim faithfulness (LLM judge) | 0.6905 |
| NLI–judge agreement | 0.5905 |
| Citation precision (vs CUAD gold spans) | 0.2589 |
| Citation recall (vs CUAD gold spans) | 0.9976 |
| Citation IoU (char offsets) | 0.2587 |

## Per-question

| Question (contract :: clause) | Claims | Faith (NLI) | Faith (Judge) | Agree | Cit P | Cit R | Cit IoU |
|---|---|---|---|---|---|---|---|
| ACME_SUPPLY_AGREEMENT::Cap On Liability | 5 | 0.8000 | 0.8000 | 1.0000 | 0.4556 | 1.0000 | 0.4556 |
| ACME_SUPPLY_AGREEMENT::Governing Law | 4 | 0.0000 | 0.7500 | 0.2500 | 0.0825 | 1.0000 | 0.0825 |
| ACME_SUPPLY_AGREEMENT::Renewal Term | 5 | 0.8000 | 0.8000 | 1.0000 | 0.1425 | 1.0000 | 0.1425 |
| ACME_SUPPLY_AGREEMENT::Termination For Convenience | 4 | 0.5000 | 1.0000 | 0.5000 | 0.3000 | 1.0000 | 0.3000 |
| GLOBEX_LICENSE::Anti-Assignment | 5 | 0.2000 | 0.4000 | 0.8000 | 0.4182 | 1.0000 | 0.4182 |
| GLOBEX_LICENSE::Governing Law | 4 | 0.0000 | 0.7500 | 0.2500 | 0.1500 | 1.0000 | 0.1500 |
| GLOBEX_LICENSE::License Grant | 3 | 1.0000 | 0.3333 | 0.3333 | 0.2636 | 0.9831 | 0.2624 |
