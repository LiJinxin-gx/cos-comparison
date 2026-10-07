# Core Identity: Extraction–Generation Inverse & Closure (v0.5.3)

## One-screen summary

- **One relation.** Every task is a local comparison of neighbouring patches.
- **Two directions.** Forward is *extraction* (data → relation graph); reverse is
  *generation* (relation graph → data). They are inverses over that relation.
- **One criterion.** *Closure*: extract the generated data and you must get the
  same relation graph.

```
        extract (forward)
   data ───────────────────────►  relation graph
    ▲                              · parts   = equality classes
    │                              · grammar = displacement connectors
    └───────────────────────  generate (reverse)
              continuous: multiplicative fixed point
              discrete : replay only grounded connectors
```

The discrete case is the binary limit of the continuous one (equality is the
discrete special case of cosine), so a single mechanism covers tensors and
symbol sequences. Composition is **partial** — a *groupoid*: parts combine only
where connection heads match a recorded morphism.

---

## Concise derivation

### 1. The local measurement

For two neighbouring patches `a`, `b`, set `A = ‖a‖²`, `B = ‖b‖²`, `C = a·b`.

```
cos    = C / √(AB)          direction only (invariant to independent scale)
mod    = 2√(AB) / (A + B)   magnitude only
cosmod = 2C / (A + B)       combined
```

They obey the exact identity **`cosmod = cos · mod`**. Equality of symbols is
the binary limit of `cos`, which is why continuous and discrete data are the
same problem.

### 2. Why extraction is many-to-one

The forward map `F` (a local comparison / box stencil) maps a field `x` to a
response `r = F x`. `F` has a non-trivial null space (measured dimension
≥ 208 for the 2-D stack; pooling contracts it further). Many fields therefore
share one response: extraction collapses them, and generation must choose one.

### 3. Reverse, continuous — a fixed point

With a non-negative local stencil, iterate multiplicatively

```
x ← x · (Fᵀ r) / (Fᵀ F x)
```

At a fixed point `Fᵀ(F x) = Fᵀ r`, so the field's response matches the target.
The update is local and deterministic (no weights, no randomness). Because the
fixed point is not unique over the null space, a *target relation graph* (more
connectors) is what pins a specific instance.

### 4. Reverse, discrete — grounded connectors

Symbols get deterministic equality classes; **connectors are recorded, not
inferred** — an edge `a → b` exists only where that adjacency was observed.
Generation replays recorded edges and joins parts only at matching connection
heads. Free composition is disallowed (it is the groupoid's partial composition).

### 5. The criterion

Re-extract the generated data. Continuous: the response residual must vanish.
Discrete: every adjacency must be a recorded connector. When the target is
partial, the correct answer is the *possibility set* (or an abstain), never a
guessed single member.

---

## Open-module platform

The reference build (`cos_platform.py`, exploration workspace) runs standalone
and imports as a module, with I/O separated from processing. Matched I/O pairs
("配套对"), verified symmetric at runtime:

| pair | mapping |
|-------|---------|
| `pack_field` / `unpack_field` | continuous 2-D field ↔ bytes (lossless) |
| `fit_symbols` / `decode_symbols` | discrete symbols ↔ integer ids (UnitMap) |

Processing functions take a tensor / sequence and return generated data with a
closure verdict and residuals: `run_continuous(field)`, `run_discrete(sequence)`.

---

## Verified results

| Task | Result | note |
|------|--------|------|
| Continuous closure (inline ring) | mean\|dE\| ≈ 1e-6, max ≈ 2e-6 | exact fixed point |
| Digit closure via **formal backend** | mean\|dE\| = 4e-6, max = 2.5e-4 | not self-certified |
| Discrete grounded recombination | grounded connectors 100%, novel | partial composition |
| Face recognition (ORL, multi-scale) | total 65.8%, abstain 25.8%, **on-classified 88.8%** | CNN reference 90% |
| Synthetic video (n-dimensional passive) | recognition 95.8%, abstain 4.2%, on-classified 100%; next-frame 98.9% | controlled data |
| Modern English text (prior-free n-grams) | total 58.3%, abstain 20.7%, on-classified 73.5% | no linguistic priors |

**Data ceiling (corrected).** Raw nearest-neighbour on MNIST with the full 60k
training set: L1 = 94.70%, L2 = 95.73% (full-10k references ≈ 96.9% / ≈ 97%);
irreducible ambiguity ≈ **3–5%**, superseding an earlier small-sample estimate.
Fashion-MNIST raw L2 ceiling = 84.13%.

---

## Evidence-backed boundaries

1. **Local closure is necessary, not sufficient.** Every local trigram recorded
   (each from a different context) can still be globally invalid ("thespace");
   cross-class image mixes close but are malformed.
2. **Generation needs a target graph.** At branch points without one it is
   underdetermined — return the possibility set or abstain ("three-valued: do
   not guess").
3. **Arrangement is kept.** The more intrinsic / translation-invariant a
   descriptor, the weaker its discrimination; position accumulates through local
   relative displacements and is not deleted.
4. **Saturation explains the adjacent-scale language failure** (measured ≈ 0.965);
   the repeating unit — recurring n-grams across documents — is discovered, not
   assumed.
5. **Discrimination and generation differ**: topic uses part distributions;
   coherent generation replays the recorded sequential connector map.

---

## Forensic method

Change / binary fields use a fixed `[0,1]` range. Uniform fields and near-zero
residuals are visually ambiguous under auto-scaling (all-1 ≠ all-0 look alike;
~1e-16 residuals get amplified), so heat-maps are always paired with per-position
numbers; the numbers are the authority.
