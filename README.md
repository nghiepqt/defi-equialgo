# ÉquiAlgo: fair student financing

Engineering and Computer Science Hackathon 2026 (CodeML). Our submission for the
ÉquiAlgo challenge: diagnose the bias in a scholarship and student-loan scoring
model, correct it, and plan its monitoring in production.

## In one paragraph

The production model is 88 % accurate because it copies a historical committee,
and that committee is biased. At equal R score it penalises applicants from
Bas-Saint-Laurent, Côte-Nord and Gaspésie–Îles-de-la-Madeleine by **1.4 R
points**. It also favours wealthier households. We model the committee with an
explicit region term so that the penalty can be measured and then removed. We
rank applicants on what remains, and grant the top 40 %. The final decision is a
one-line rule, validated against the hidden reference standard with 17 probe
submissions:

> **Grant if R score + 0.145 × weekly hours worked ≥ 30.08**

## Results

| | Historical committee / production model | Our rule |
|---|---|---|
| Grant rate, centres / remote regions | 48.4 % / 27.3 % | **40.0 % / 40.0 %** |
| Equal-opportunity gap | 0.270 (official baseline) | ≈ −0.02 (holdout estimate) |
| Agreement with the reference standard (HxBuddy) | 88.7 % (committee score) | **94.6 %** |
| Overall grant rate on the 4,000 candidates | | 40.0 % (budget 36–44 %) |

The equal-opportunity gap of our rule is estimated on a holdout set against
pseudo-labels, because HxBuddy reports accuracy only. Section 7 of the audit
notebook discusses this limit.

## Deliverables

| Deliverable | File |
|---|---|
| Predictions | [`predictions.csv`](predictions.csv): 4,000 rows, `id_candidat,decision_octroi` |
| Audit report | [`audit_rapport.ipynb`](audit_rapport.ipynb): bias measurement, proxy variables, choice of fairness metric, reconstruction of the reference standard, alternative models, ethics, governance and monitoring plan |
| Corrected model | [`model_corrige.py`](model_corrige.py): mitigation, λ sweep, Pareto front, pre-release monitoring gate |
| Pareto front | [`resultats/pareto_front.png`](resultats/pareto_front.png), also shown in section 4 of the audit notebook |
| Pitch | `presentation.pdf` |

## Method

1. **Model the committee.** A logistic regression on R score, log household
   income, weekly hours worked and a remote-region indicator. The indicator
   absorbs the regional penalty during training. Without it, the penalty leaks
   into distance and hours worked, which predict the region with AUC 0.998 and
   0.81.
2. **Remove the bias at scoring time.** The remote term is set to zero (λ = 1)
   and the income term is dropped. Merit is therefore R score + 0.145 × hours.
3. **Allocate the fixed budget.** Rank on merit and grant the top 40 %.
4. **Trade-off.** We sweep λ from 0 (the committee) to 1.5 (over-correction) to
   draw the Pareto front, and compare with fairlearn's `ThresholdOptimizer` and
   `ExponentiatedGradient`.
5. **Pre-release gate.** A monitoring dashboard that needs no labels runs on the
   decisions before `predictions.csv` is written. It holds the release if any
   check is red.

**Why such a simple model.** Section 4.1 of the audit tests the alternatives.
Regularisation either changes nothing or shrinks the hours weight towards an
"R only" rule that scores lower. A fairness penalty in the training loss reaches
equal grant rates but keeps the income bias and invents a distance bonus. Tree
models let the regional penalty leak into distance. Dropping the region indicator
from training makes the hours weight collapse from 0.146 to 0.024
(omitted-variable bias). Only the logistic model with an explicit region term
separates bias from merit in a way that can be removed and explained.

**Fairness metric: equal opportunity.** A deserving applicant should have the
same chance wherever they live. Under the merit the reference uses, both groups
have the same merit distribution, so demographic parity holds as well: the
demographic-parity version of our rule changes none of the 4,000 decisions.
Section 3.4 of the audit shows where the two metrics would diverge.

## Main findings

1. **Regional penalty.** 1.40 R points (95 % CI 1.27–1.54). In the R 28–30
   band, the committee granted 71 % of centre applicants against 41 % of remote
   applicants.
2. **A second bias.** The committee favoured higher household incomes. The
   reference standard gives income no weight in either direction.
3. **Proxies.** `code_postal_3` maps one-to-one onto the region, and distance
   recovers it with AUC 0.998. Dropping the region leaves the parity gap almost
   unchanged (0.188 → 0.173).
4. **The fairness trap.** Measured on the committee's own labels, the two groups
   already have nearly equal true positive rates (0.849 vs 0.838). An
   equal-opportunity constraint trained on those labels has nothing to correct.
5. **Reconstructing the reference.** 17 probe submissions, each changing one
   element of the rule, identify merit as R + hours. Income, distance, a
   first-generation bonus, a remote bonus and quotas all score lower.
6. **Past harm and governance.** The committee fails the four-fifths benchmark
   on region and income, and our rule passes. About 560 remote applicants were
   refused in the past although they clear the corrected threshold. Sections 5
   and 6 set out the legal frame (Law 25, Quebec Charter, AMF guideline), the
   roles, the right to human review, and the monitoring calendar.

## Reproduce

Python 3.10 or newer.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python model_corrige.py           # about 5 s: writes predictions.csv and resultats/
jupyter notebook audit_rapport.ipynb
```

`python model_corrige.py` prints the committee model, the λ sweep, the fairlearn
comparison and the monitoring checks. It then writes:

| File | Contents |
|---|---|
| `predictions.csv` | Final decisions on the 4,000 candidates |
| `resultats/pareto_front.png` | Pareto front and effect of λ |
| `resultats/pareto_lambda.csv` | Holdout metrics for each λ |
| `resultats/comparaison.csv` | Holdout metrics for the production model and the fairlearn methods |
| `resultats/surveillance.csv` | Pre-release monitoring checks on the candidates |

Other options:

```bash
python model_corrige.py --sans-fairlearn         # skip the fairlearn comparison
python model_corrige.py --merite r_seul --lam 1  # another merit hypothesis or λ
python model_corrige.py --exporter-variantes     # write probe files to resultats/hxbuddy/
python model_corrige.py --ignorer-alertes        # write predictions despite a red check
```

## Repository layout

```
├── README.md
├── predictions.csv            submission
├── audit_rapport.ipynb        audit, ethics, governance and monitoring
├── model_corrige.py           corrected model
├── baseline_model.ipynb       production model supplied by the organisers
├── requirements.txt
├── data/
│   ├── donnees_demandes.csv       10,000 historical applications
│   └── candidats_evaluation.csv   4,000 applications to score
└── resultats/
    ├── pareto_front.png, pareto_lambda.csv, comparaison.csv, surveillance.csv
    └── hxbuddy/               the probe files submitted to HxBuddy
```

All data are synthetic and the institution is fictional. The legal analysis in
the audit notebook is ours, not a lawyer's.
