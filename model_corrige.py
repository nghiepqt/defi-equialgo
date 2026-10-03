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

Evaluation caveat: the reference standard is hidden. On the holdout we measure
against pseudo-labels drawn from the neutralised committee model, i.e. under
an assumption on what merit is. The `ecart_conditionnel_r` metric does not
depend on that assumption: it compares grant rates of the two groups at equal
R score.

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
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

def ecart_conditionnel_r(pred, groupe, cote_r, n_bins=8):
    """Grant-rate gap between groups at equal R score, weighted by bin size."""
    bins = pd.qcut(cote_r, n_bins, labels=False, duplicates='drop')
    ecarts, poids = [], []
    for b in np.unique(bins):
        m = bins == b
        c, r = m & (groupe == 'Center'), m & (groupe == 'Remote')
        if c.sum() and r.sum():
            ecarts.append(pred[c].mean() - pred[r].mean())
            poids.append(m.sum())
    return float(np.average(ecarts, weights=poids))


def mesurer(pred, y_ref, y_hist, groupe, cote_r):
    pred, y_ref, y_hist = map(np.asarray, (pred, y_ref, y_hist))
    out = {'taux_octroi': pred.mean()}
    for g in ('Center', 'Remote'):
        m = groupe == g
        out[f'taux_{g}'] = pred[m].mean()
        out[f'tpr_{g}'] = pred[m & (y_ref == 1)].mean()
    out['ecart_parite'] = out['taux_Center'] - out['taux_Remote']
    out['ecart_eo'] = out['tpr_Center'] - out['tpr_Remote']
    out['ecart_conditionnel_r'] = ecart_conditionnel_r(pred, groupe, np.asarray(cote_r))
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


def tracer(balayage, autres, lam_choisi, merite, chemin):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Signed gap on the x axis: positive favours centres, negative favours
    # remote regions. Pareto optimality is computed on its absolute value.
    tous = pd.concat([balayage, autres])
    front = tous[front_pareto(tous['ecart_eo'].abs(), tous['accord_reference'])]
    ax1.scatter(front['ecart_eo'], front['accord_reference'], s=160, facecolors='none',
                edgecolors='0.55', lw=1.5, zorder=1, label='Pareto optimal')
    ax1.axvline(0, color='black', lw=0.8)

    ax1.plot(balayage['ecart_eo'], balayage['accord_reference'], 'o-',
             color='tab:blue', label='Neutralisation, sweep of λ')
    for _, r in balayage.iterrows():
        ax1.annotate(f"{r['lam']:g}", (r['ecart_eo'], r['accord_reference']),
                     textcoords='offset points', xytext=(4, 4), fontsize=7, color='tab:blue')
    choisi = balayage[np.isclose(balayage['lam'], lam_choisi)]
    if len(choisi):
        ax1.scatter(choisi['ecart_eo'], choisi['accord_reference'], s=260,
                    facecolors='none', edgecolors='tab:red', lw=2, zorder=5,
                    label=f'Chosen point (λ={lam_choisi:g})')

    # Methods landing on the same point share one label.
    for (x, y), bloc in autres.groupby(['ecart_eo', 'accord_reference'], sort=False):
        noms = list(bloc.index)
        nom = noms[0] if len(noms) == 1 else f"{noms[0]} (+{len(noms) - 1} identical)"
        if 'RF' in nom:
            m, c = 'X', 'black'
        else:
            m, c = ('s' if 'Exp' in nom else 'D' if 'Thresh' in nom else '^'), 'tab:orange'
        ax1.scatter(x, y, marker=m, color=c, s=60, zorder=4)
        ax1.annotate(nom, (x, y), textcoords='offset points', xytext=(5, -10), fontsize=7)
    ax1.set_xlabel('Equal-opportunity gap vs pseudo-reference (TPR centres - TPR remote)')
    ax1.set_ylabel('Agreement with pseudo-reference')
    ax1.set_title(f'Fairness / utility trade-off (holdout, merit = {merite})')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='lower left', fontsize=8)

    ax2.plot(balayage['lam'], balayage['taux_Center'], 'o-', label='Grant rate, centres')
    ax2.plot(balayage['lam'], balayage['taux_Remote'], 'o-', label='Grant rate, remote')
    ax2.plot(balayage['lam'], balayage['ecart_conditionnel_r'], 's--',
             label='Gap at equal R score')
    ax2.plot(balayage['lam'], balayage['accord_comite'], '^:', color='0.4',
             label='Agreement with historical committee')
    ax2.axhline(0, color='black', lw=0.8)
    ax2.axvline(lam_choisi, color='tab:red', lw=1, ls='--')
    ax2.set_xlabel('λ, share of the regional penalty removed')
    ax2.set_title('Effect of λ')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


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
    groupe, cote_r = groupe_region(test), test['cote_r_equivalent'].to_numpy()

    # 2. Sweep of lambda.
    lignes = []
    for lam in LAMBDAS:
        pred = allouer(modele.score(test, lam=lam, **reglage), args.taux)
        lignes.append({'lam': lam, **mesurer(pred, y_ref, y_hist, groupe, cote_r)})
    balayage = pd.DataFrame(lignes)

    # 3. Comparison points.
    autres = {'RF production': rf_production(train, test)}
    if not args.sans_fairlearn:
        print('fairlearn comparison (ExponentiatedGradient takes a little while)...')
        autres.update(comparaison_fairlearn(train, test))
    autres = pd.DataFrame({
        nom: mesurer(p, y_ref, y_hist, groupe, cote_r) for nom, p in autres.items()
    }).T

    colonnes = ['taux_octroi', 'taux_Center', 'taux_Remote', 'ecart_parite', 'ecart_eo',
                'ecart_conditionnel_r', 'accord_reference', 'accord_comite']
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
