# ÉquiAlgo : un financement étudiant équitable

Hackathon Génie et Informatique 2026. Défi de 24 heures.

Une institution financière québécoise évalue les demandes de bourses
d'excellence et de prêts d'études avec un modèle d'apprentissage automatique. Il
atteint 88 % d'exactitude. Un audit interne a constaté qu'il accorde une bourse
à 48,4 % des candidat·es de Montréal et de la Capitale-Nationale, contre 27,3 %
au Bas-Saint-Laurent, sur la Côte-Nord et en
Gaspésie–Îles-de-la-Madeleine.

La cote R moyenne est de 27,3 dans les régions éloignées et de 28,0 dans les
grands centres. Cela explique une partie de l'écart de 21 points. Le reste est
inexpliqué.

Votre mandat : diagnostiquer le biais, le corriger, et proposer un plan de
surveillance en production.

Toutes les données sont synthétiques. L'institution est fictive.

## Installation

```bash
git clone <URL_DE_VOTRE_DEPOT>
cd defi-equialgo
python3 -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook baseline_model.ipynb
```

Python 3.10 ou plus récent. Exécutez le carnet une fois avant toute
modification. Il entraîne le modèle en production, le mesure et l'audite.

## Fichiers

| Fichier | Lignes | Contenu |
|---|---|---|
| `data/donnees_demandes.csv` | 10 000 | demandes historiques, avec `decision_octroi` |
| `data/candidats_evaluation.csv` | 4 000 | demandes à évaluer, sans étiquette |
| `baseline_model.ipynb` | | modèle en production, métriques, audit d'équité |

### Colonnes

| Colonne | Signification |
|---|---|
| `id_candidat` | identifiant, format `C000000` |
| `cote_r_equivalent` | performance académique, équivalent cote R, de 15 à 40 |
| `programme_etudes` | programme d'études, 5 catégories |
| `region_administrative` | variable sensible, région administrative du Québec |
| `code_postal_3` | trois premiers caractères du code postal |
| `revenu_familial_estime` | revenu annuel brut du ménage |
| `heures_travail_semaine` | heures travaillées par semaine pendant les études |
| `distance_domicile_campus_km` | domicile au campus, en km |
| `premiere_generation_universitaire` | 1 si première personne de la famille à l'université |
| `decision_octroi` | cible, 1 accordé, 0 refusé |

Les noms de régions sont écrits sans accents et sans espaces autour du trait
d'union : `Montreal`, `Capitale-Nationale`, `Bas-Saint-Laurent`, `Cote-Nord`,
`Gaspesie-Iles-de-la-Madeleine`.

## Contraintes

**Enveloppe fixe.** Votre taux d'octroi sur les 4 000 candidat·es d'évaluation
doit se situer entre 36 % et 44 %. Hors de cette plage, la section technique
vaut zéro.

**`decision_octroi` n'est pas la cible.** Le jury note contre un étalon de
référence construit indépendamment du comité historique. Vous ne l'avez pas.
Cette colonne consigne ce que le comité a fait, et c'est le comité qui est
audité.

**Supprimer `region_administrative` ne fonctionne pas.** Retirer cette colonne
fait passer l'écart de parité de 0,188 à 0,181. Retirer aussi le code postal
l'amène à 0,173. La distance, les heures travaillées, le revenu familial et le
code postal portent tous de l'information régionale. La section 5 du carnet le
mesure.

## Livrables

Un dépôt GitHub, public ou partagé avec les juges, contenant :

- `predictions.csv` à la racine. Deux colonnes, 4 000 lignes plus l'en-tête,
  valeurs 0 ou 1. La dernière cellule du carnet en écrit un exemple valide.
- `audit_rapport.ipynb`. La mesure du biais, vos métriques d'équité avec leur
  justification, et les variables proxys que vous avez trouvées.
- `model_corrige.py` ou `.ipynb`. Votre solution d'atténuation, avec un
  graphique du front de Pareto pour plusieurs réglages de la contrainte.
- `presentation.pdf`. Le support d'un pitch de cinq minutes.

```csv
id_candidat,decision_octroi
C000042,1
C000117,0
```

## Notation

| Section | Points | Évalué par |
|---|---|---|
| Rigueur du diagnostic | 25 | le jury |
| Solution technique | 35 | le correcteur automatique |
| Gouvernance et éthique | 25 | le jury |
| Pitch et qualité du code | 15 | le jury |

Les 35 points automatiques, tous deux mesurés contre l'étalon de référence
caché :

- Équité, 20 points. Proportion de l'écart d'égalité des chances du modèle de
  base qui est refermée. Cet écart de départ vaut 0,270.
- Utilité, 15 points. Concordance avec l'étalon de référence, sur une échelle
  allant d'un tirage aléatoire respectant l'enveloppe à une répartition
  parfaite.

Les deux valent zéro si la contrainte d'enveloppe est brisée.

## Notes

`fairlearn.postprocessing.ThresholdOptimizer` ajuste les seuils de décision
après l'entraînement et s'exécute en quelques secondes.
`fairlearn.reductions.ExponentiatedGradient` réentraîne sous contrainte et prend
plusieurs minutes.

La parité démographique et l'égalité des chances ne peuvent pas tenir ensemble
quand les deux groupes ont des profils différents. Choisissez-en une et
préparez-vous à défendre ce choix.

Un modèle unique ne fait pas un front de Pareto. Balayez la contrainte d'équité
et tracez les résultats.

Des mentors sont disponibles en tout temps.
