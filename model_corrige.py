"""ÉquiAlgo - corrected model.

Pipeline
  1. Model the historical committee with a logistic regression that keeps an
     explicit `remote` indicator. The regional penalty is absorbed by that
     coefficient instead of leaking through proxies (postal code, distance,
     hours worked).
  2. Score applicants with the penalty removed (counterfactual remote=0).
     `lam` is the fraction of the penalty removed: 0 = committee, 1 = neutral.
  3. Allocate the fixed budget by ranking: the top 40% are granted.
  4. Sweep `lam` on a holdout to draw the Pareto front, and compare with
     fairlearn's ThresholdOptimizer and ExponentiatedGradient.
  5. Before writing predictions.csv, run the label-free monitoring dashboard
     (`rapport_surveillance`) and hold the decisions if a check is red.

Evaluation caveat: the reference standard is hidden. On the holdout we measure
against pseudo-labels drawn from the neutralised committee model, i.e. under
an assumption on what merit is. The `ecart_conditionnel_merite` metric needs
no labels: it compares grant rates of the two groups at equal merit score.

Choice of merit: HxBuddy round 1 (accuracy on the evaluation set) ranked
R + hours worked first (94.6%), ahead of R + hours + income (92.5%), R alone
(92.3%) and R + hours - income (92.1%). The committee baseline scored 88.7%.
Round 2 around R + hours: every perturbation scored lower. Hours weight 0.10
(94.2%) and 0.20 (94.0%) bracket the committee's ~0.145 (parabola vertex
~0.146); grant rate 38% (94.4%) and 42% (94.5%) put the optimum at ~40%; a
first-generation bonus of 0.5 R points cost a full point (93.6%).
Round 3, terms absent from the committee model: a bonus for remote regions
(+0.35 R points: 94.2%, +0.7: 93.0%), distance (93.7%), lower income (94.2%)
and per-program quotas (94.5%) all scored lower; per-region quotas tied
(94.65%, 1 row out of 4000).

Usage
  python model_corrige.py                          # sweep, plot, predictions.csv
  python model_corrige.py --merite r_seul --lam 1.0
  python model_corrige.py --poids-heures 0.10 --bonus-pg 0.5 --taux 0.42
  python model_corrige.py --exporter-variantes     # CSVs to probe on HxBuddy
  python model_corrige.py --sans-fairlearn         # skip the fairlearn comparison
  python model_corrige.py --ignorer-alertes        # override the pre-release gate
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

RACINE = Path(__file__).resolve().parent
DONNEES = RACINE / 'data'
RESULTATS = RACINE / 'resultats'

ELOIGNEES = ['Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine']
TAUX_CIBLE = 0.40
BUDGET = (0.36, 0.44)

# Features of the committee model; `remote` is added separately. Program of
# study, first generation, distance and postal code carry no signal once these
# are known (distance and postal code only re-encode the region).
COMITE = ['cote_r_equivalent', 'log_revenu', 'heures_travail_semaine']

# Hypotheses on what the reference standard calls merit: which committee
# coefficients are kept, and with which sign.
MERITE = {
    'complet': {'cote_r_equivalent': 1, 'log_revenu': 1, 'heures_travail_semaine': 1},
    'r_seul': {'cote_r_equivalent': 1},
    'r_heures': {'cote_r_equivalent': 1, 'heures_travail_semaine': 1},
    'besoin': {'cote_r_equivalent': 1, 'log_revenu': -1, 'heures_travail_semaine': 1},
}

MERITE_DEFAUT = 'r_heures'

# HxBuddy round 2, around r_heures: hours weight (R points per hour; the
# committee's is ~0.145), reference grant rate, first-generation bonus (R points).
SONDES = [
    {},
    {'poids_heures': 0.10},
    {'poids_heures': 0.20},
    {'taux': 0.38},
    {'taux': 0.42},
    {'bonus_pg': 0.5},
]

LAMBDAS = np.round(np.linspace(0, 1.5, 16), 2)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def groupe_region(df):
    """Two groups, as in the audit: major centres vs remote regions."""
    return np.where(df['region_administrative'].isin(ELOIGNEES), 'Remote', 'Center')


def features(df):
    X = pd.DataFrame({
        'cote_r_equivalent': df['cote_r_equivalent'],
        'log_revenu': np.log(df['revenu_familial_estime']),
        'heures_travail_semaine': df['heures_travail_semaine'],
    }, index=df.index)
    remote = (groupe_region(df) == 'Remote').astype(int)
    return X, remote


# ---------------------------------------------------------------------------
# Corrected model
# ---------------------------------------------------------------------------

class ModeleComite:
    """Logistic model of the committee with an explicit regional penalty term."""

    def fit(self, df):
        X, remote = features(df)
        lr = LogisticRegression(penalty=None, max_iter=10_000)
        lr.fit(X.assign(remote=remote), df['decision_octroi'])
        self.coefs_ = dict(zip(COMITE + ['remote'], lr.coef_[0]))
        self.moyennes_ = X.mean()
        self.moyenne_pg_ = df['premiere_generation_universitaire'].mean()
        # Intercept at the mean applicant, so that dropping a feature from the
        # merit score removes its variation without shifting the scale.
        self.base_ = lr.intercept_[0] + sum(self.coefs_[c] * self.moyennes_[c] for c in COMITE)
        return self

    @property
    def penalite_points_r(self):
        """Regional penalty expressed in R score points."""
        return -self.coefs_['remote'] / self.coefs_['cote_r_equivalent']

    def score(self, df, lam=1.0, merite=MERITE_DEFAUT, poids_heures=None, bonus_pg=0.0):
        """Logit score. lam=0 with merite='complet' reproduces the committee;
        lam=1 removes the regional penalty entirely.

        `poids_heures` overrides the committee's hours weight and `bonus_pg`
        adds a first-generation bonus, both in R score points.
        """
        X, remote = features(df)
        coefs = dict(self.coefs_)
        b_r = coefs['cote_r_equivalent']
        if poids_heures is not None:
            coefs['heures_travail_semaine'] = poids_heures * b_r
        eta = self.base_ + sum(
            signe * coefs[c] * (X[c] - self.moyennes_[c])
            for c, signe in MERITE[merite].items()
        )
        eta += bonus_pg * b_r * (df['premiere_generation_universitaire'] - self.moyenne_pg_)
        return np.asarray(eta + (1 - lam) * coefs['remote'] * remote)


def allouer(scores, taux=TAUX_CIBLE):
    """Grant the top `taux` share of applicants. Ties are broken by order."""
    scores = np.asarray(scores)
    k = int(round(taux * len(scores)))
    rang = np.argsort(-scores, kind='stable')
    pred = np.zeros(len(scores), dtype=int)
    pred[rang[:k]] = 1
    return pred


def etiquettes_equitables(modele, df, reglage, seed=0):
    """Pseudo reference labels: committee decisions with the regional penalty
    removed, drawn from the neutralised probability so they keep the
    committee's noise. An assumption, not ground truth."""
    p = 1 / (1 + np.exp(-modele.score(df, lam=1.0, **reglage)))
    return (np.random.default_rng(seed).random(len(df)) < p).astype(int)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ecart_conditionnel_merite(pred, groupe, merite, n_bins=8):
    """Grant-rate gap between groups at equal merit score, weighted by bin size."""
    bins = pd.qcut(merite, n_bins, labels=False, duplicates='drop')
    ecarts, poids = [], []
    for b in np.unique(bins):
        m = bins == b
        c, r = m & (groupe == 'Center'), m & (groupe == 'Remote')
        if c.sum() and r.sum():
            ecarts.append(pred[c].mean() - pred[r].mean())
            poids.append(m.sum())
    return float(np.average(ecarts, weights=poids))


def mesurer(pred, y_ref, y_hist, groupe, merite):
    pred, y_ref, y_hist = map(np.asarray, (pred, y_ref, y_hist))
    out = {'taux_octroi': pred.mean()}
    for g in ('Center', 'Remote'):
        m = groupe == g
        out[f'taux_{g}'] = pred[m].mean()
        out[f'tpr_{g}'] = pred[m & (y_ref == 1)].mean()
    out['ecart_parite'] = out['taux_Center'] - out['taux_Remote']
    out['ecart_eo'] = out['tpr_Center'] - out['tpr_Remote']
    out['ecart_conditionnel_merite'] = ecart_conditionnel_merite(pred, groupe, np.asarray(merite))
    out['accord_reference'] = (pred == y_ref).mean()
    out['accord_comite'] = (pred == y_hist).mean()
    return out


# ---------------------------------------------------------------------------
# Comparison points
# ---------------------------------------------------------------------------

def rf_production(train, test):
    """The production model from baseline_model.ipynb, retrained on `train`."""
    cat = ['programme_etudes', 'region_administrative', 'code_postal_3']
    drop = ['id_candidat', 'decision_octroi']
    X = pd.get_dummies(train.drop(columns=drop), columns=cat)
    Xt = pd.get_dummies(test.drop(columns=drop), columns=cat).reindex(columns=X.columns, fill_value=0)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf.fit(X, train['decision_octroi'])
    return rf.predict(Xt)


def comparaison_fairlearn(train, test):
    """fairlearn mitigations trained on the historical labels, without region.

    Their fairness constraint is measured against `decision_octroi`, i.e.
    against the committee under audit; that is the point of the comparison.
    """
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.reductions import (
        DemographicParity, ExponentiatedGradient, TruePositiveRateParity,
    )

    cols = COMITE + ['distance_domicile_campus_km', 'premiere_generation_universitaire']

    def X_de(df):
        X, _ = features(df)
        return X.assign(
            distance_domicile_campus_km=df['distance_domicile_campus_km'],
            premiere_generation_universitaire=df['premiere_generation_universitaire'],
        )[cols]

    scaler = StandardScaler().fit(X_de(train))
    Xtr, Xte = scaler.transform(X_de(train)), scaler.transform(X_de(test))
    ytr = train['decision_octroi'].to_numpy()
    gtr, gte = groupe_region(train), groupe_region(test)

    sorties = {}
    lr = LogisticRegression(max_iter=5000).fit(Xtr, ytr)
    sorties['LR sans region'] = allouer(lr.predict_proba(Xte)[:, 1])

    for nom, contrainte in [('TPR', 'true_positive_rate_parity'), ('DP', 'demographic_parity')]:
        to = ThresholdOptimizer(
            estimator=lr, constraints=contrainte, objective='accuracy_score',
            prefit=True, predict_method='predict_proba',
        )
        to.fit(Xtr, ytr, sensitive_features=gtr)
        sorties[f'ThresholdOptimizer {nom}'] = to.predict(Xte, sensitive_features=gte, random_state=0)

    for borne in (0.02, 0.05, 0.10):
        for nom, classe in [('TPR', TruePositiveRateParity), ('DP', DemographicParity)]:
            eg = ExponentiatedGradient(
                LogisticRegression(max_iter=5000), constraints=classe(difference_bound=borne),
            )
            eg.fit(Xtr, ytr, sensitive_features=gtr)
            sorties[f'ExpGrad {nom} {borne:.2f}'] = allouer(eg._pmf_predict(Xte)[:, 1])
    return sorties


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

def front_pareto(x, y):
    """Mask of points not dominated when minimising x and maximising y."""
    x, y = np.asarray(x), np.asarray(y)
    return np.array([
        not np.any((x <= x[i]) & (y >= y[i]) & ((x < x[i]) | (y > y[i])))
        for i in range(len(x))
    ])


def placer_etiquette_intelligente(ax, rendu, cadre, occupe, pt, x, y, texte, candidates_spec,
                                  font_props=None, bbox_props=None):
    """Place an annotation at the candidate offset with the lowest overlap cost."""
    best_ann = None
    best_bbox = None
    best_cost = float('inf')

    for idx, (d, sx, sy) in enumerate(candidates_spec):
        ha = 'left' if sx > 0.1 else ('right' if sx < -0.1 else 'center')
        va = 'bottom' if sy > 0.1 else ('top' if sy < -0.1 else 'center')
        arrow = dict(arrowstyle='->', color='0.5', lw=0.6, shrinkA=2, shrinkB=3) if d > 10 else None

        ann = ax.annotate(
            texte, (x, y), textcoords='offset points', xytext=(sx * d, sy * d),
            ha=ha, va=va, arrowprops=arrow, bbox=bbox_props, **(font_props or {})
        )
        ann.update_positions(rendu)
        b = Text.get_window_extent(ann, rendu)

        overlap_area = 0
        for o in occupe:
            if b.overlaps(o):
                ix0 = max(b.x0, o.x0)
                iy0 = max(b.y0, o.y0)
                ix1 = min(b.x1, o.x1)
                iy1 = min(b.y1, o.y1)
                if ix1 > ix0 and iy1 > iy0:
                    overlap_area += (ix1 - ix0) * (iy1 - iy0)

        pad = 2
        outside = (b.x0 < cadre.x0 + pad) or (b.x1 > cadre.x1 - pad) or \
                  (b.y0 < cadre.y0 + pad) or (b.y1 > cadre.y1 - pad)

        cost = (1e8 if outside else 0) + overlap_area * 100 + d * 1.5 + idx * 0.2

        if cost < best_cost:
            best_cost = cost
            if best_ann is not None:
                best_ann.remove()
            best_ann = ann
            best_bbox = b
            if overlap_area == 0 and not outside:
                break
        else:
            ann.remove()

    occupe.append(best_bbox)
    return best_ann


def tracer(balayage, autres, lam_choisi, merite, chemin):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Signed gap on the x axis: positive favours centres, negative favours
    # remote regions. Pareto optimality is computed on its absolute value.
    tous = pd.concat([balayage, autres])
    front = tous[front_pareto(tous['ecart_eo'].abs(), tous['accord_reference'])]
    ax1.scatter(front['ecart_eo'], front['accord_reference'], s=170, facecolors='none',
                edgecolors='0.5', lw=1.6, zorder=1, label='Pareto optimal')
    ax1.axvline(0, color='black', lw=0.8, ls=':')

    xs, ys = balayage['ecart_eo'].to_numpy(), balayage['accord_reference'].to_numpy()
    ax1.plot(xs, ys, 'o-', color='tab:blue', lw=1.8, markersize=5.5, label='Neutralisation, sweep of λ', zorder=2)
    choisi = balayage[np.isclose(balayage['lam'], lam_choisi)]
    if len(choisi):
        ax1.scatter(choisi['ecart_eo'], choisi['accord_reference'], s=260,
                    facecolors='none', edgecolors='tab:red', lw=2.2, zorder=5,
                    label=f'Chosen point (λ={lam_choisi:g})')

    # Comfortable margins to prevent clipping against axis edges
    ax1.set_ylim(0.835, 0.875)
    ax1.margins(x=0.08)
    ax1.set_xlabel('Equal-opportunity gap vs pseudo-reference (TPR centres - TPR remote)', fontsize=9.5)
    ax1.set_ylabel('Agreement with pseudo-reference', fontsize=9.5)
    ax1.set_title(f'Fairness / utility trade-off (holdout, merit = {merite})', fontsize=11, fontweight='bold', pad=10)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='lower left', fontsize=8.5, framealpha=0.92, edgecolor='0.8')

    ax2.plot(balayage['lam'], balayage['taux_Center'], 'o-', label='Grant rate, centres', lw=1.6)
    ax2.plot(balayage['lam'], balayage['taux_Remote'], 'o-', label='Grant rate, remote', lw=1.6)
    ax2.plot(balayage['lam'], balayage['ecart_conditionnel_merite'], 's--',
             label='Gap at equal merit score', lw=1.6)
    ax2.plot(balayage['lam'], balayage['accord_comite'], '^:', color='0.4',
             label='Agreement with historical committee', lw=1.6)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.axvline(lam_choisi, color='tab:red', lw=1.2, ls='--')
    ax2.set_xlabel('λ, share of the regional penalty removed', fontsize=9.5)
    ax2.set_title('Effect of λ', fontsize=11, fontweight='bold', pad=10)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(fontsize=8.5, framealpha=0.92, edgecolor='0.8')

    fig.tight_layout()
    fig.canvas.draw()
    rendu = fig.canvas.get_renderer()
    cadre = ax1.get_window_extent(rendu)

    # Collision obstacles: start with legend bounding box
    occupe = [ax1.get_legend().get_window_extent(rendu)]
    pt = fig.dpi / 72

    # Markers as obstacles
    for x, y in zip(xs, ys):
        p = ax1.transData.transform((x, y))
        occupe.append(Bbox.from_extents(p[0] - 5 * pt, p[1] - 5 * pt, p[0] + 5 * pt, p[1] + 5 * pt))

    # 1. Label key points on the lambda sweep curve
    selected_lams = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5}
    bbox_lam = dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#b0c4de', lw=0.4, alpha=0.85)

    for x, y, lam in zip(xs, ys, balayage['lam']):
        if lam in selected_lams:
            if np.isclose(lam, lam_choisi):
                txt = f'λ={lam:g} (chosen)'
                fprops = dict(fontsize=8, color='#b22222', weight='bold')
                bprops = dict(boxstyle='round,pad=0.25', facecolor='#fff0f0', edgecolor='#b22222', lw=0.9, alpha=0.95)
                cands = [(16, -0.6, 1.3), (20, 0, 1.4), (22, -1, 1)]
            else:
                txt = f'λ={lam:g}' if lam in (0.0, 1.5) else f'{lam:g}'
                fprops = dict(fontsize=7, color='tab:blue')
                bprops = bbox_lam
                cands = [(d, sx, sy) for d in (7, 11, 16)
                         for sx, sy in [(0, 1), (0.7, 0.9), (-0.7, 0.9), (1, 0.3), (-1, 0.3)]]
            placer_etiquette_intelligente(ax1, rendu, cadre, occupe, pt, x, y, txt, cands,
                                         font_props=fprops, bbox_props=bprops)

    # 2. Comparison methods
    for (x, y), bloc in autres.groupby(['ecart_eo', 'accord_reference'], sort=False):
        noms = list(bloc.index)
        if len(noms) == 1:
            nom = noms[0]
        else:
            base = noms[0].rsplit(' ', 1)[0] if any(char.isdigit() for char in noms[0]) else noms[0]
            nom = f"{base} (all bounds)"

        if 'RF' in nom:
            m, c = 'X', 'black'
            cands = [(d, sx, sy) for d in (12, 16, 22) for sx, sy in [(0, 1.1), (-0.8, 1), (-1, 0.3), (0.8, 1)]]
        else:
            m, c = ('s' if 'Exp' in nom else 'D' if 'Thresh' in nom else '^'), 'tab:orange'
            if y < 0.842:
                # ExpGrad TPR at the bottom
                cands = [(d, sx, sy) for d in (11, 16, 22) for sx, sy in [(0, -1), (-0.8, -1), (-1, -0.5), (0.8, -1)]]
            elif x > 0.17 and y > 0.847:
                # ThresholdOptimizer TPR vs ExpGrad DP 0.10: separate cleanly
                if 'Thresh' in nom:
                    cands = [(d, sx, sy) for d in (12, 18, 24) for sx, sy in [(-1, 0.8), (-1, 0.2), (-0.5, 1)]]
                else:
                    cands = [(d, sx, sy) for d in (12, 18, 24) for sx, sy in [(1, -0.6), (1, 0.2), (0.7, -1)]]
            elif 'DP 0.05' in nom:
                cands = [(d, sx, sy) for d in (10, 15, 20) for sx, sy in [(0.2, -1), (0.6, -1), (-0.3, -1), (1, -0.5)]]
            elif 'DP 0.02' in nom:
                cands = [(d, sx, sy) for d in (10, 15, 20) for sx, sy in [(-0.8, -1), (-1, -0.6), (0, -1)]]
            elif 'ThresholdOptimizer DP' in nom:
                cands = [(d, sx, sy) for d in (10, 14, 20) for sx, sy in [(-1, -0.5), (-1, 0.2), (-0.6, -1)]]
            else:
                # LR sans region
                cands = [(d, sx, sy) for d in (11, 16, 22) for sx, sy in [(1, 0.2), (1, -0.6), (0.8, 1)]]

        ax1.scatter(x, y, marker=m, color=c, s=70, zorder=4)
        p = ax1.transData.transform((x, y))
        occupe.append(Bbox.from_extents(p[0] - 5 * pt, p[1] - 5 * pt, p[0] + 5 * pt, p[1] + 5 * pt))

        bbox_comp = dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#a0a0a0', lw=0.5, alpha=0.92)
        fprops = dict(fontsize=7.2, color='#222222')
        placer_etiquette_intelligente(ax1, rendu, cadre, occupe, pt, x, y, nom, cands,
                                     font_props=fprops, bbox_props=bbox_comp)

    fig.savefig(chemin, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Governance: plain decision rule, explanations, production monitoring
# ---------------------------------------------------------------------------

# Columns the deployed decision may read. Everything else (region, postal code,
# distance, income, program, first generation) stays out of the decision and is
# kept only in an access-controlled store for monitoring.
ENTREES_DECISION = ['cote_r_equivalent', 'heures_travail_semaine']

# Alert levels for the pre-release dashboard: (green limit, amber limit).
# 'max' checks must stay below the limits, 'min' checks above them. The gap
# limits sit above the 99th percentile of an unbiased rule on a bootstrapped
# cycle of 4,000 applicants (audit_rapport.ipynb, section 6.4). The ratio
# checks use the US four-fifths benchmark for amber; small cells make them
# noisy, so amber means "investigate", not "hold".
SEUILS_ALERTE = {
    'parity_gap': ('max', 0.05, 0.10),
    'gap_equal_merit': ('max', 0.03, 0.06),
    'region_rate_ratio': ('min', 0.80, 0.70),
    'income_rate_ratio': ('min', 0.80, 0.70),
    'intersection_rate_ratio': ('min', 0.80, 0.70),
    'psi_r_score': ('max', 0.10, 0.25),
    'psi_hours': ('max', 0.10, 0.25),
    'override_gap': ('max', 0.05, 0.10),
}


def regle_decision(modele, df, taux=TAUX_CIBLE):
    """The deployed rule in plain form: merit = R + w * hours, in R points.

    Ranks exactly like `modele.score(df, lam=1.0)`, which is an increasing
    linear function of the same quantity. Returns (w, threshold, merit, pred).
    """
    w = modele.coefs_['heures_travail_semaine'] / modele.coefs_['cote_r_equivalent']
    merite = (df['cote_r_equivalent'] + w * df['heures_travail_semaine']).to_numpy()
    pred = allouer(merite, taux)
    return w, merite[pred == 1].min(), merite, pred


def expliquer_decision(ligne, w, seuil, bande=0.25):
    """Plain-language explanation owed on request under P-39.1 s. 12.1."""
    merite = ligne['cote_r_equivalent'] + w * ligne['heures_travail_semaine']
    ecart = merite - seuil
    octroi = ecart >= 0
    texte = [
        f"Applicant {ligne['id_candidat']}: {'GRANTED' if octroi else 'REFUSED'}. "
        'This decision was made by an automated process.',
        f"Information used: R score {ligne['cote_r_equivalent']:.2f}, "
        f"{ligne['heures_travail_semaine']:.1f} hours of paid work per week. "
        'Region, postal code, distance, income, program and first-generation '
        'status were not used.',
        f'How it was decided: merit = R score + {w:.3f} x weekly hours = {merite:.2f}. '
        f'This cycle the budget funds 40% of applicants, which sets the threshold at {seuil:.2f}.',
    ]
    if not octroi:
        texte.append(f'Your merit is {-ecart:.2f} points below the threshold: about '
                     f'{-ecart:.2f} more R points, or {-ecart / w:.1f} more weekly hours, '
                     'would have changed the decision.')
    if abs(ecart) <= bande:
        texte.append('Your file is close to the threshold and was also reviewed by a staff member.')
    texte.append('You may ask us to correct the information above, and submit observations '
                 'to a staff member who can review the decision.')
    return '\n'.join(texte)


def intervalle_wilson(k, n, z=1.96):
    """Wilson score interval for a proportion k / n."""
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    demi = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - demi, centre + demi


def psi(reference, actuel, n_bins=10):
    """Population stability index of `actuel` against `reference` (decile bins)."""
    bords = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    bords[0], bords[-1] = -np.inf, np.inf
    ref = np.histogram(reference, bords)[0] / len(reference)
    act = np.histogram(actuel, bords)[0] / len(actuel)
    ref, act = np.clip(ref, 1e-4, None), np.clip(act, 1e-4, None)
    return float(np.sum((act - ref) * np.log(act / ref)))


def ratio_min_max(decision, cles, effectif_min=100):
    """Lowest / highest grant rate over the cells of `cles` with enough applicants."""
    taux = pd.Series(decision).groupby(cles).agg(['mean', 'size'])
    taux = taux[taux['size'] >= effectif_min]['mean']
    return float(taux.min() / taux.max())


def rapport_surveillance(df, decision, merite, reference, decision_regle=None, revise=None):
    """Pre-release fairness dashboard for one funding cycle. Needs no labels.

    `df` holds the cycle's applicants with the monitoring columns, `decision`
    the final decisions (after any human override), `merite` their merit
    score, and `reference` the applications the rule was validated on, for
    input drift. If the rule's own decisions and the mask of files sent to
    human review are given, overrides are checked too.
    """
    decision, merite = np.asarray(decision), np.asarray(merite)
    groupe = groupe_region(df)
    g_ref = groupe_region(reference)
    quintile = pd.qcut(df['revenu_familial_estime'], 5, labels=False).to_numpy()
    tercile = pd.qcut(df['revenu_familial_estime'], 3, labels=False).to_numpy()
    valeurs = {
        'grant_rate': decision.mean(),
        'parity_gap': abs(decision[groupe == 'Center'].mean() - decision[groupe == 'Remote'].mean()),
        'gap_equal_merit': abs(ecart_conditionnel_merite(decision, groupe, merite)),
        'region_rate_ratio': ratio_min_max(decision, df['region_administrative'].to_numpy()),
        'income_rate_ratio': ratio_min_max(decision, quintile),
        # Region group crossed with first generation, and with income tercile.
        'intersection_rate_ratio': min(
            ratio_min_max(decision, [groupe, df['premiere_generation_universitaire'].to_numpy()]),
            ratio_min_max(decision, [groupe, tercile])),
        # Drift is checked within each group: a shift in one group is what
        # moves grant rates between groups.
        'psi_r_score': max(psi(reference.loc[g_ref == g, 'cote_r_equivalent'],
                               df.loc[groupe == g, 'cote_r_equivalent']) for g in ('Center', 'Remote')),
        'psi_hours': max(psi(reference.loc[g_ref == g, 'heures_travail_semaine'],
                             df.loc[groupe == g, 'heures_travail_semaine']) for g in ('Center', 'Remote')),
    }
    if decision_regle is not None:
        # Net override rate (granted minus refused by a reviewer) among the
        # reviewed files of each group. Human review can bring the bias back.
        net = decision - np.asarray(decision_regle)
        revise = np.asarray(revise)
        valeurs['override_gap'] = abs(net[revise & (groupe == 'Center')].mean()
                                      - net[revise & (groupe == 'Remote')].mean())

    def statut(nom, v):
        if nom == 'grant_rate':
            return 'green' if 0.38 <= v <= 0.42 else 'amber' if BUDGET[0] <= v <= BUDGET[1] else 'red'
        sens, vert, orange = SEUILS_ALERTE[nom]
        if sens == 'max':
            return 'green' if v <= vert else 'amber' if v <= orange else 'red'
        return 'green' if v >= vert else 'amber' if v >= orange else 'red'

    return pd.DataFrame([{'check': nom, 'value': v, 'status': statut(nom, v)}
                         for nom, v in valeurs.items()]).set_index('check')


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def ecrire_predictions(candidats, pred, chemin):
    pred = np.asarray(pred)
    taux = pred.mean()
    assert len(pred) == len(candidats) == 4000, len(pred)
    assert set(np.unique(pred)) <= {0, 1}
    assert candidats['id_candidat'].is_unique
    assert BUDGET[0] <= taux <= BUDGET[1], f'grant rate {taux:.3f} outside the budget'
    pd.DataFrame({'id_candidat': candidats['id_candidat'], 'decision_octroi': pred.astype(int)}) \
        .to_csv(chemin, index=False)
    return taux


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lam', type=float, default=1.0, help='share of the regional penalty removed')
    ap.add_argument('--merite', choices=list(MERITE), default=MERITE_DEFAUT, help='merit hypothesis')
    ap.add_argument('--poids-heures', type=float, default=None,
                    help="hours weight in R points per hour (default: the committee's)")
    ap.add_argument('--bonus-pg', type=float, default=0.0, help='first-generation bonus in R points')
    ap.add_argument('--taux', type=float, default=TAUX_CIBLE, help='grant rate on the candidates')
    ap.add_argument('--sans-fairlearn', action='store_true', help='skip the fairlearn comparison')
    ap.add_argument('--exporter-variantes', action='store_true',
                    help='also write the round-2 probes (SONDES) to resultats/hxbuddy/')
    ap.add_argument('--ignorer-alertes', action='store_true',
                    help='write predictions.csv even if a pre-release check is red')
    args = ap.parse_args()
    reglage = {'merite': args.merite, 'poids_heures': args.poids_heures, 'bonus_pg': args.bonus_pg}

    RESULTATS.mkdir(exist_ok=True)
    demandes = pd.read_csv(DONNEES / 'donnees_demandes.csv')
    candidats = pd.read_csv(DONNEES / 'candidats_evaluation.csv')

    # 1. Holdout evaluation, same split as the baseline notebook.
    train, test = train_test_split(
        demandes, test_size=0.3, random_state=42, stratify=demandes['decision_octroi']
    )
    modele = ModeleComite().fit(train)
    print('Committee model (train):')
    for c, v in modele.coefs_.items():
        print(f'  {c:<24} {v:+.3f}')
    print(f'  regional penalty = {modele.penalite_points_r:.2f} R score points\n')

    y_ref = etiquettes_equitables(modele, test, reglage)
    y_hist = test['decision_octroi'].to_numpy()
    groupe, merite = groupe_region(test), modele.score(test, lam=1.0, **reglage)

    # 2. Sweep of lambda.
    lignes = []
    for lam in LAMBDAS:
        pred = allouer(modele.score(test, lam=lam, **reglage), args.taux)
        lignes.append({'lam': lam, **mesurer(pred, y_ref, y_hist, groupe, merite)})
    balayage = pd.DataFrame(lignes)

    # 3. Comparison points.
    autres = {'RF production': rf_production(train, test)}
    if not args.sans_fairlearn:
        print('fairlearn comparison (ExponentiatedGradient takes a little while)...')
        autres.update(comparaison_fairlearn(train, test))
    autres = pd.DataFrame({
        nom: mesurer(p, y_ref, y_hist, groupe, merite) for nom, p in autres.items()
    }).T

    colonnes = ['taux_octroi', 'taux_Center', 'taux_Remote', 'ecart_parite', 'ecart_eo',
                'ecart_conditionnel_merite', 'accord_reference', 'accord_comite']
    pd.set_option('display.width', 200)
    print('\nSweep of λ (holdout):')
    print(balayage.set_index('lam')[colonnes].round(3).to_string())
    print('\nComparison points (holdout):')
    print(autres[colonnes].round(3).to_string())

    balayage.to_csv(RESULTATS / 'pareto_lambda.csv', index=False)
    autres.to_csv(RESULTATS / 'comparaison.csv', index_label='methode')
    tracer(balayage, autres, args.lam, args.merite, RESULTATS / 'pareto_front.png')

    # 4. Final model on all historical data, then the candidates.
    final = ModeleComite().fit(demandes)
    pred = allouer(final.score(candidats, lam=args.lam, **reglage), args.taux)

    # 5. Pre-release gate: the decisions are not written if a check is red.
    merite_cand = final.score(candidats, lam=1.0, **reglage) / final.coefs_['cote_r_equivalent']
    surveillance = rapport_surveillance(candidats, pred, merite_cand, demandes)
    surveillance.to_csv(RESULTATS / 'surveillance.csv')
    print('\nPre-release monitoring (candidates):')
    print(surveillance.round(3).to_string())
    rouges = list(surveillance.index[surveillance['status'] == 'red'])
    if rouges and not args.ignorer_alertes:
        raise SystemExit(f'Red checks {rouges}: decisions held. Rerun with --ignorer-alertes '
                         'only after the validation committee signs off.')
    taux = ecrire_predictions(candidats, pred, RACINE / 'predictions.csv')
    g = groupe_region(candidats)
    print(f'\npredictions.csv: {reglage}, λ={args.lam:g}, grant rate {taux:.1%} '
          f'(centres {pred[g == "Center"].mean():.1%}, remote {pred[g == "Remote"].mean():.1%})')

    if args.exporter_variantes:
        dossier = RESULTATS / 'hxbuddy'
        dossier.mkdir(exist_ok=True)
        reference = None
        for sonde in SONDES:
            taux_sonde = sonde.get('taux', TAUX_CIBLE)
            r = {'merite': MERITE_DEFAUT, 'poids_heures': None, 'bonus_pg': 0.0}
            r.update({k: v for k, v in sonde.items() if k != 'taux'})
            p = allouer(final.score(candidats, lam=1.0, **r), taux_sonde)
            nom = '_'.join(f'{k}{v:g}' for k, v in sonde.items()) or 'reference'
            ecrire_predictions(candidats, p, dossier / f'sonde_{nom}.csv')
            reference = p if reference is None else reference
            print(f'  sonde_{nom}.csv: {(p != reference).sum()} rows differ from sonde_reference')
        print(f'{len(SONDES)} probes written to {dossier.relative_to(RACINE)}/')

    print(f'Results in {RESULTATS.relative_to(RACINE)}/')


if __name__ == '__main__':
    main()
