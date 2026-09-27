# Structural-Priority Text Model: Large-Scale Distributed Training Log

Exploration log (v62-v65). Structure-first, unsupervised, no deep-learning
primitives (no weights/gradients/argmax/temperature). Native cos_comparison
components only: `UnitMap`, `window_units`, `most_common`.

## Core principle applied to text

- **Continuous mapping hierarchy isolation (A<->B)**, information emerges from
  local comparison; structure first, details on demand.
- Text primitives (cos is for numeric tensors; on discrete symbols use):
  - **passive** = run-fold breakpoints (where consecutive units change);
  - **active** = `window_units` + `most_common` (high-frequency repeated units).
- **Absolute locality**: every fragment is an independent worker process.
- **No loss, layered split-store**: L0 details, L1 boundaries, L2 high-freq
  templates; layers store different info, combined they are complete.

## Key lesson: small corpus is an information-theory limit

On an 871-token toy corpus the passive distribution looked flat (no
boundaries) and token emergence looked coincidental. Scaling to 42k-2M tokens
removed the limit: passive distribution became healthy and high-frequency
tokens emerged for real. Symbols are polysemous/conventional and need enough
data to reveal regularities.

## Results

| Version | Corpus | Tokens | Workers | High-freq tokens | Generation |
|---|---|---|---|---|---|
| v62 | English | 216k | 4 | of the / in the / to be / mr darcy | "the same" |
| v63 | mixed EN+ZH | 2.04M | 6 | of the / Chinese multi-char tokens | mixed (junk) |
| v64 | cleaned EN + ZH | 424k + 1.31M | 6 | the same / Chinese multi-char tokens | EN "the same" |
| v65 | EN + reasoning | 424k | 6 | 42 rules (of->the, the->same...) | "the same" |

Data cleaning mattered: strip Gutenberg headers/footers, drop image paths /
filenames, separate EN and ZH tokenization (EN by word, ZH by character).

## Reasoning engine (v65)

High-frequency bigrams become a rule graph `reason -> [results]`
(`window_units(local_size=2)` + `most_common`). Interactive generation walks
the rule graph from a seed; when no rule applies it stops (three-valued: it
derives, or returns unknown -- it never guesses). Example:
`the -> same / other / common / law / world ...`.

## Open problems

- EN `of the / in the` appear joined because `window_units` folds adjacent
  tokens into one unit; this is intended token emergence, not junk.
- Generation is rule-graph chaining (structural), not semantic coherence.
- Chinese tokenization is character-level; multi-char words (e.g. proper nouns) emerge
  as bigram units.
