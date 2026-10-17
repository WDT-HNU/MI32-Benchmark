# Dataset index

The benchmark covers MI3, MI2, and P300. Traditional, deep, and Riemannian methods are evaluated against task data contracts; method families are not dataset categories.

## Task contracts

| Task | Labels | Main recorded release | Typical input |
|---|---|---|---|
| MI3 | `left_upper` / `right_upper` / `non_upper` | common-32 v4: 8 sources, 230 subjects, 97,608 trials | `[32,750] @ 250 Hz` |
| MI2 | Left / Right | deep-model view: 230 subjects, 48,804 trials; Binary MI contract: 18 sources, 389 subjects, 59,216 trials | `[32,750] @ 250 Hz` |
| P300 | Target / NonTarget | 15 reported eligible datasets; 16-channel, 0–600 ms ERP contract | `[16,121] @ 200 Hz` |

## Dataset cards

| Card | Purpose |
|---|---|
| [MI32 common-32 v4](mi32/DATASET_CARD.md) | current MI3 release and its MI2 derived view |
| [Binary MI](binary_mi/DATASET_CARD.md) | 18-source left/right contract used by the Riemannian archive and reflected in the traditional snapshot |
| [P300](p300/DATASET_CARD.md) | reviewed 16-channel P300 harmonization and LOGO contract |

## Split identities matter

The MI2 deep-model view and the 18-source Binary MI release define the same task but are not the same frozen dataset. Likewise, P300 traditional, deep, and Riemannian results use 873, 102, and 85 folds respectively. Keep release and split identity attached to every number; do not place scores into a paired ranking merely because their task label matches.
