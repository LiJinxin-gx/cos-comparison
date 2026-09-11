# Exploration Tests (v0.5.0)

Exploratory testing summary — a general method (local comparison + hierarchical
isolation + threshold mapping + parallelism) validated across image, audio and
text tasks, plus learning mechanisms, Agent frameworks and field web-automation
experiments.

The exploration modules live in the `explore/` directory at the project root:
core module (`group_hierarchical.py`) and demo
scripts (`demo_speech.py`, `demo_text.py`, `demo_image.py`). They are
distributed as exploration examples in the source distribution (they are NOT
installed as a package — the wheel carries no `explore` module; run them from
the source tree). See [atomic_contrast_matching.md](atomic_contrast_matching.md)
for the latest atomic contrast point matching results.

> Test platform: Intel Core Ultra 5 125H (14C/18T) + Intel Arc Graphics
> (112 CUs, driver 31.0.101.5382); Windows 11 x64; Python 3.14.6; numpy
> 2.5.2; pyopencl 2026.1.4.  CPU-only experiments exclude the GPU.

## General Method

```
data -> numeric tensor (1D/2D) -> core hierarchical isolation
        (multi-kernel active probes / passive comparison)
     -> threshold layering (high-percentile / linear anchor mapping)
     -> block activation / label maps -> prototype matching / clustering
parallelism: free-threaded (no-GIL Python 3.14 build) + ThreadPoolExecutor
```

## Atomic Contrast Point Matching (PCML v10.37-v10.42)

Unsupervised structure-first matching using binary contrast point sets and
Jaccard similarity, instead of global feature vectors and cosine similarity.
Key result: **47.1%** on 30-class speech recognition, exceeding supervised
methods (22.6%).

See [atomic_contrast_matching.md](atomic_contrast_matching.md) for full
tables and methodology.

| Method | Train/class | Accuracy |
|--------|------------|----------|
| Supervised Fisher baseline | 100 | 22.6% |
| Unsupervised K-means | 80 | 7.2% |
| Atomic Jaccard KNN | 500 | 42.9% |
| **Multi-scale + Hierarchical** | **500** | **47.1%** |

Core module in `explore/`: `group_hierarchical.py`.

## Core Algorithm Validation

### Face Recognition (67 ID-photo training -> 2 life-photo validation)

| Version | Method | Subject A | Subject B |
|---------|--------|-----------|-----------|
| v4 | Hierarchical isolation (active Fourier kernels -> high-percentile quantize -> block activation) | 67/67 | 66/66 (1.0000) |
| v5 | Dual-path cross + RGB + part-level | 67/67 | 66/66 (1.0000) |
| v6 | 5 evidence sources + EventBinds logic + cross-degree | 67/67 | 66/66 (1.0000, c=5) |
| v7 | Multi-scale (64/48/32) hierarchical isolation | 67/67 | 66/66 (1.0000) |

Results consistent across all four versions (Subject B full score) — stable matching.

### Music Genre (GTZAN, 10 genres x 100 tracks)

| Feature | Accuracy |
|---------|----------|
| Waveform energy blocks | 11.0% |
| Waveform texture (energy/zero-crossing/variance) | 13.0% |
| Full-track segments + autocorrelation beat | 14.0% |
| **FFT spectrogram** (segmented FFT -> log magnitude -> 2D -> hierarchical) | **24.0%** |
| Multi-FFT concatenation | 21.0% (diluted) |

Parallel (8 threads + pydll): 1000 tracks in 5.9s.

### Handwritten Digits (MNIST, 60000/10000)

| Method | Accuracy |
|--------|----------|
| Binary block activation (50000 train) | 68.2% |
| Interval labels + reflex adoption learning | 48.1% (mechanism validated) |
| **Linear anchor mapping + weighted difference** | **73.6%** (rotated 15 deg: 65.6%, robust) |
| Distributed training (8 workers + iterative rounds) | 73.7% |
| Active feedback learning (error-sample prototype reinforcement) | 73.1% -> 73.6% |

### Text Clustering (heterogeneous project texts: docs/tests/History)

| Method | doc-code similarity |
|--------|---------------------|
| Word frequency + idf (unified vocabulary) | 0.0555 |
| **Hierarchical label learning** (encode -> 2-gram -> new labels -> layered mapping) | **0.0061** (10x discrimination) |

20 Newsgroups classification: 14.4% (random baseline 5%);
clustering limited by real data connectivity.

### Captcha (PIL-generated, 32 character classes x 4 chars)

| Dataset | Method | char / whole |
|---------|--------|--------------|
| arial clean | Weighted difference + sliding window | **99.8% / 99.0%** |
| arial clean | Distributed complementary hints (6 workers) | **100.0% / 100.0%** |
| arial clean | Unit data template storage (multi-level matching) | 99.8% / 99.0% |
| hard (rotation + noise lines) | Single trainer / complementary hints | 13.5% / 14.2% |

### Learning Mechanism Experiments

| Mechanism | Validation |
|-----------|------------|
| Reflex adoption learning (Monitor/Trigger + confidence gate + depth limit anti-pollution) | fully working |
| Active feedback (error samples reinforce prototypes) | MNIST 73.1% -> 73.6% |
| Complementary-hint distributed (non-voting: min evidence per worker) | captcha 99.8% -> 100.0% |
| Multi-dimension template merge generation (consensus on same points + abstract-level arbitration on conflicts) | generated data closed-loop recognition 100% |

### Autonomous Agent (built from the seven-layer modules)

Autonomous captcha recognition Agent (Sense/Brain/ControlFlowDriver/Action/
Memory/Reflex):
- Recognition 100.0% (complementary hints triggered 400 times)
- Async action layer (ExecuterDriver) wrote 100 history records
  (SQLite thread-independent connections)
- Full loop: perceive -> cognize -> decide -> act -> memorize -> reflect

## Data Generation

- Template material stored in SQLite (templates table)
- Multi-dimension merge generation (unit/material/category dims;
  conflicts resolved by abstraction level)
- Replace operation = read + generate + overwrite (UPDATE DB)
- 100 generated captchas recognized at 100% (generator usable for
  data augmentation)

## Video Abstraction & Generation Training

Method (v3, reusing 10 local + downloaded videos, 22545 frames):

```
L0 frames -> L1 cos features (passive/active response + downsampling)
          -> L2 scene-segment prototypes (segment mean = same points)
          -> L3 video summary (prototype sequence)
prototypes (same points) + contrast points (segment differences)
          stored in SQLite; generation = high-level concretization
          via contrast-point weighted aggregation (learned weights)
training: joint across 3 videos (36 frame pairs), coordinate descent
          + parallel candidate evaluation, free-threaded workers
```

Result vs the v2 baseline standard (identical frame-pair selection):

| Metric | v2 (guided interpolation) | v3 (learned contrast aggregation) |
|--------|---------------------------|-----------------------------------|
| Generation improvement | -1.6% | **+20.2%** (training MAE +47.1%) |
| Cross-video generalization | n/a | **+6.4%** (unseen video) |
| Data | 1 video | 10 videos / 22545 frames in SQLite |

Learned weights: struct=-0.600, edge=-0.500, motion=-0.125 (subtracting
motion/structure interference is optimal for long-gap interpolation of
animation-style content). Parallel build ~9.5 min for 10 videos.

## Agent Framework (von Neumann & Memory)

### Behavior-Composition Memory Agent (formal framework)

Von-Neumann structure: behaviors stored as data in the same database as
knowledge (definition / target / args-schema / defaults / enabled /
usage statistics), executed by the instruction cycle:

```
fetch (behavior from DB) -> decode (defaults merge)
     -> reflect-execute (interface.api CallDict + importlib loader)
     -> write back (runs history + statistics)
```

Formal framework (historical, now in experiment area) (pure stdlib + cos_comparison):

- `BehaviorMemory`: DatabaseMemory with injected DatabaseToolWrap
  (check_same_thread=False) - single shared connection + write lock,
  `:memory:` and parallel-safe
- `BehaviorExecutor`: `CallDict` dispatch; default loader resolves
  `'module.func'` targets via importlib reflection (injectable
  `loader_func` = delegation); composite behaviors execute their DB
  step sequence in order
- `BehaviorComposer` / `run_parallel` (ExecuterDriver worker) /
  `Agent` assembly with ControlFlowDriver + Monitor
- Dynamic reconfiguration via DB only, zero code change:
  `configure(name, "target", ...)` swaps implementation,
  `configure(name, "enabled", 0)` disables,
  `configure(name, "defaults", ...)` changes parameters
- External behavior modules (e.g. selenium-backed) are injected
  implementations; the module self-test uses inline test behaviors - no network, no third-party deps

### Generic Executor (atomic instruction triples)

Formal evolution (historical, now in experiment area) — fixed code runs any
logic encoded in the DB as instruction triples:

```
instructions table: (pc, func, args, kwargs)   # (func-name, positional,
                                                #  keyword) atomic unit
run_program(name): fetch(DB) -> decode($N result refs + reflection)
                 -> execute(plugin func) -> write back to registers
composite embed:  func = 'program:<sub>'  -> recursive sub-program
plugin style:     func = any 'module.func' reflected at runtime
error isolation:  per-instruction failure returns {"error": ...}
```

- `$N` refs pass previous results (positional/keyword args alike)
- Composite embedding uses register copies (sub-program isolation)
- Prompt-driven stage planning (`prompt_plan`) maps external prompts to
  behavior sequences for ordinary and specified tasks with one fixed code

### Reflex & Self-Correction (REFLECT/CORRECT instructions)

Von-Neumann engine extension (`<experiment-dir>/von_surf_engine.py`):

```
BEHAVE  dst behavior args            execute external behavior
REFLECT dst target behavior          inspect quality (reflection)
CORRECT dst target reflection        fix by reflection (self-correction)
```

- Entry reflection: relevance / trusted source / noise / completeness /
  duplicate scoring per entry; correction removes <60 scores
- Paper reflection: section completeness / empty-data markers / length;
  correction appends a `[REFLEX-CORRECTED]` record section
- Judged improvement: noise 16.7% -> 0.0%, paper completeness 100%,
  composite accuracy 86/100 (24 -> 10 kept entries)

### Feedback-Driven Search

`derive_queries` derives new search directions from data already in the
database (no new code, just instruction data):

- Frequent relevant terms (noise/used-word filtered) -> new queries
- View-coverage gaps (VIEW_KEYWORDS classes with no hits) -> gap queries
- Targeted re-search -> merge -> re-analyze -> paper

Result: cluster groups 2 -> 5, composite accuracy 90/100 (instruction
program 23 lines all stored in DB).

### Hierarchical Isolation Memory (continuous mapping + layer driving)

`<experiment-dir>/agent_hier_behaviors.py` + `von_hier_engine.py` — the
project methodology applied to memory storage, NOT weight compression:

```
L1 entry layer   (raw, full retention incl. unrelated data)
L2 topic layer   (continuous mapping: feature vec vs proto cos score,
                  threshold isolation: score>thr joins proto else new)
L3 view layer    (view classification + coverage gaps)
```

- **On-demand extraction**: only needs-relevant entries are abstracted
  upward; unrelated data stays isolated in L1 (no pollution of high layers)
- **Detail preservation**: abstracted entries keep diff vectors + detail
  terms (no detail loss)
- **Two-way layer driving**: bottom-up (L1->L2->L3 aggregation) and
  top-down (L3 gaps -> needs signals -> drive L2 mapping and L1 search);
  isolation is NOT permanent — needs evolution re-evaluates isolated
  entries (no missed details)
- **Decoupled search/arrange**: producers (parallel search workers) write
  L1 only; consumers read L1 write L2/L3; cooperation via need signals

## Field & Web Automation Tests

Real-world web automation experiments against anti-bot and login-fortress
sites. All driven by the generic executor / behavior-agent framework with
zero code change between tasks.

> **Security & Responsible Use Notice**
>
> These experiments validate the framework's capability to operate on
> modern web platforms with anti-automation defenses. Specific target
> platforms, CSS selectors, URL patterns, and verification-barrier
> fingerprints are deliberately generalized below. The demonstrated
> techniques (profile-cookie reuse, in-page JS extraction, verification
> detection + human handoff) are for defensive research and framework
> validation only. Do not use this framework to attack, overload, or
> bypass access controls of any service without explicit authorization.
> Automated access must comply with each platform's terms of service and
> applicable law.

### 1. Web-Search Knowledge Base (search engine + online encyclopedia)

Pipeline: general search engine results + online encyclopedia fetch (headless
Edge via selenium, Selenium Manager auto-driver, custom UA against
anti-bot, 8-thread parallel on free-threaded Python) -> field extraction
(infobox dt/dd XPath) -> SQLite knowledge base with provenance
(source/URL/timestamp).

| Task | Result |
|------|--------|
| Chemical elements | 118/118 (symbol & atomic number 100% complete after dt/dd extraction + authority-table validation, 17 symbol mis-picks corrected) |
| ISO 639 languages | 188 codes with Chinese names |

Generated documents: `elements_encyclopedia.md`, `language_codes.md`.
Open encyclopedia unreachable (blocked), cloud storage DNS-blocked;
encyclopedia + search engine used as primary sources.

### 2. Firefox Logged-In Platform Fetch (login fortress)

`<experiment-dir>/agent_platform_behaviors.py` + `run_platform_program.py` —
login-fortress scenario, driven by the generic executor (13 instruction
triples in DB):

- Reuses the user's Firefox profile (copies cookies.sqlite/prefs.js/
  permissions.sqlite) to carry login state through headless Firefox
- Behaviors: login-check (multi-site + retry, Unicode-star account
  mask regex), article-fetch, posts-collect, data-collect, report-generate
- Result: login confirmed (account mask), 3 encyclopedia articles +
  2 community boards' real hot posts stored into the knowledge base with a
  generated markdown report
- Note: the platform masks accounts with full-width stars (U+FF0A) — the
  detection regex must cover `[\*\uFF0A\u2022\u2217]`

### 3. Community Platform Agent: Search + Profile Modification (verification fortress)

`<experiment-dir>/community_agent.py` + `community_diagnose.py` — a focused online-community
agent that breaks through a major online community's multi-layer verification
(slider captcha / login gate / rate limit / sensitive-action re-auth) and
performs search, profile read/write, and post content collection.

#### Fortress breakthrough comparison

| Browser | Profile | Verification hits | Search | Profile modify |
|---------|---------|-------------------|--------|----------------|
| Edge (fresh) | none | 4 (slider + 3x login) | 0 results | blocked |
| **Firefox (logged-in copy)** | **profile copy** | **0** | **10 results** | **success (verified)** |

Method: copy the user's active Firefox profile
(`%APPDATA%\Mozilla\Firefox\Profiles\*.default-release`, ~1500 files,
excluding `parent.lock`) and launch Selenium Firefox with `-profile`.
Login cookies (platform passport) carry over — no slider, no login gate.
geckodriver downloaded manually (Selenium Manager's GitHub fetch
failed in this environment).

#### Results

- Login confirmed: a logged-in test account (account metadata extracted)
- Search (test keyword): 10 results extracted via in-page JS executor
  (static CSS selectors failed on the platform's SPA markup; `execute_script`
  scanning by platform-specific partial-class selectors and post-link URL
  patterns is robust)
- Profile read: username / intro / account-age / region all extracted
  via platform-specific profile-field selectors
- Profile modify: intro changed from default placeholder to a test bio
  string — **verified on next run** (the read-back returned the new value,
  confirming the write persisted server-side)
- Post content: 3 full posts (body + replies) collected via
  platform-specific post-body selector fallback chain

#### Verification detector

`VerificationDetector` classifies 6 barrier types by URL + body keywords +
iframe scan: `captcha_slider`, `login_required`, `sms_verify`,
`rate_limit`, `content_audit`, `captcha_iframe`. On detection the agent
pauses for human-in-the-loop resolution (`wait_for_verification`, 120s
timeout) then resumes — no captcha bypassing, only detection + handoff.

Anti-automation: `excludeSwitches=["enable-automation"]`, custom UA,
per-character typing with 50-150 ms jitter, 1-3 s random human delays.

## Datasets & Code Locations

Paths below use placeholders: `<data-dir>` = local dataset root,
`<experiment-dir>` = the experiment workspace root (a local venv directory).
Web-automation script names are generalized to avoid identifying specific
target platforms; local filenames may differ.

- Datasets: `<data-dir>/` (MNIST / 20 Newsgroups / captcha*);
  `<data-dir>/video/` (7 CPU/OS MP4, 142.6 MB);
  `<data-dir>/video_dl/` (3 downloaded public samples:
  Big Buck Bunny / Jellyfish / Sintel, Blender CC movies)
- Code: `<data-dir>/code/` (face_v4-v7 / gtzan / mnist_train /
  news_train / text_cluster / text_learn / captcha_test / captcha_active /
  captcha_complement / captcha_templates / captcha_generate / captcha_agent /
  mnist_diff / mnist_dist / mnist_reflex / gen_captcha)
- Video training: `<experiment-dir>/video_agent_gen_v3.py` (output:
  `<experiment-dir>/video_output_v3/`, includes `video_agent_gen.db`)
- Web knowledge base: `<experiment-dir>/agent_web_knowledge.py`
  (+ fix/validate passes; output: `<experiment-dir>/agent_output/`)
- Behavior Agent experiments: `<experiment-dir>/behavior_agent_v1.py`,
  `behavior_agent_v2.py`, `agent_behaviors.py` (output:
  `<experiment-dir>/behavior_output/`, `<experiment-dir>/behavior_output_v2/`);
  formal framework (historical, now in experiment area) (explicit interface, no config files)
- Surf tasks (feedback / reflex / hier / platform): `von_surf_engine.py`,
  `agent_surf.py`, `agent_surf_behaviors.py`, `von_hier_engine.py`,
  `agent_hier_behaviors.py`, `agent_platform_behaviors.py`,
  `run_platform_program.py` (+ `geckodriver.exe`, `surf_config*.json`,
  `task_prompt.txt`; output: `surf_output/`)
- Community Platform Agent (verification fortress): `community_agent.py`,
  `community_diagnose.py` (+ `geckodriver.exe`, `firefox_profile_copy/`;
  output: `community_output/` with search JSON, post contents, action &
  verification logs, screenshots)
- Environment: a free-threaded Python 3.14 venv (no-GIL build) + cp314t
  numpy/PIL/cos_comparison

## Key Methodology Findings

1. Local comparison features are effective on image/audio/text (discrimination
   quantifiable on each)
2. Threshold mapping determines discrimination: sparse high-percentile >
   equal-interval > histogram
3. Weighted difference (difference magnitude x salient-region weight) beats
   plain difference counting and cosine similarity
4. Active cognition (database knowledge applied back) works but is bounded by
   passive feature quality (GIGO)
5. Distributed complementary hints > voting > single trainer (strong-feature
   tasks)
6. Free-threaded multi-thread parallelism effective (3.4x measured; CUDA
   availability depends on hardware)
7. Behavior-as-data (von Neumann) decouples agent logic from behavior
   implementations: DB-configurable target/params/enabled and delegation
   slots allow reconfiguration without code changes
8. Authority-table validation is a reliable final check for web-scraped
   fields (corrects infobox label mismatches and navigation-text noise)
9. Reflex (REFLECT/CORRECT) improves downstream accuracy: entry-quality
   filtering cut noise 16.7% -> 0%; paper completeness restored to 100%
10. Feedback-driven search beats fixed queries: deriving directions from
    stored data (gaps + frequent terms) raised cluster groups 2 -> 5
11. Hierarchical isolation memory: demand-driven abstraction (only
    needs-related data rises; unrelated stays isolated) keeps high layers
    clean while preserving details — isolation must be re-evaluable when
    needs evolve, so high-level needs never miss low-level details
12. Login-fortress sites are accessible via user-profile cookie
    reuse (Firefox profile copy); account masks may use full-width
    Unicode stars — regex must cover them
13. SPA sites (modern community UI) defeat static CSS selectors — in-page
    `execute_script` scanning by partial-class and link-pattern is the
    robust extraction method; full-profile copy (not just cookies.sqlite)
    preserves session state across Selenium restarts, and profile-modify
    writes can be verified by read-back on a fresh launch

## External Research Assessment (Summary)

External research (kept outside the official tree; widely used public open
datasets are named for reproducibility, while specific third-party platforms
and task setups are generalized) stress-tested the general method — perception
(passive difference sensing /
active template matching) + hierarchical memory + lazy similarity-based
decision — across standard public benchmarks and internal synthetic tasks
in the image, text, retrieval and generation domains.

### Confirmed strengths

- Retrieval, matching and consistency-oriented generation are the stable
  home territory: high accuracy with millisecond-scale training, no
  gradient loop, naturally incremental learning (new memories usable at
  inference time), full interpretability and deterministic behaviour.
- Decoupled training (a sensing/generation reconstruction objective before
  decision discrimination) measurably improves downstream discrimination;
  simple lazy learning beats stacked multi-mechanism designs; low-
  dimensional signatures keep the pattern space tractable.
- The repeatedly validated design principles — information lives in
  differences; local analysis is absolute while global is relative; simple
  beats complex; reconfiguration through data rather than code — hold
  across domains and serve as a stable methodological core.

### Empirically established boundaries

- Classification strength concentrates on small, aligned, template-like
  inputs; real-world variability (position, rotation, noise) degrades
  accuracy sharply, and run-to-run stability is not yet guaranteed.
- Text tasks remain near chance for true topic discrimination, and
  generation stays at the surface level (statistical character transfer,
  no semantics); exact-value alignment across modalities is bounded by an
  information-theoretic ceiling.
- Hand-built features mostly lack class-discriminative power; abstraction
  beyond the two explicit memory levels does not emerge without manual
  design.

### Characterisation

The method is best characterised as a non-parametric memory-and-retrieval
system: strong as a transparent similarity core — a natural memory /
retrieval / interpretability layer for hybrid architectures — while the
abstract representation and compositional generalisation required for
general intelligence are not yet supplied by this approach alone.

### Next-step directions

1. Stronger input representations (learned or pre-trained embeddings) on
   top of the existing memory / decision core — the highest-leverage move
2. Learned features replacing hand-crafted ones (reconstruction-decoupled
   training)
3. Invariance-aware sensing (position / rotation / noise handling)
4. A language path via embeddings + memory retrieval, and generation by
   fragment recombination under structure constraints
5. Incremental concept formation (clustering / prototype learning) for
   the higher memory levels
6. A unified validation harness with mandatory random baselines before
   claims
## GPU Acceleration Test Data (Intel Arc)

Exploratory measurement of element-space parallel work on the local
Intel(R) Arc(TM) Graphics device (OpenCL via pyopencl; 14.3 GiB global
memory, 4 GiB max single allocation, 1024 max work-group) - the
SuperParallel element-space model maps directly onto OpenCL work items
(one kernel invocation per configured element; data/hardware transfer
stays a data-layer concern).

Test platform: Intel Core Ultra 5 125H (14C/18T) + Intel Arc Graphics
(112 CUs, driver 31.0.101.5382); Windows 11 x64; Python 3.14.6; numpy
2.5.2; pyopencl 2026.1.4.

### Saxpy benchmark (y = a*x + y, float32)

| n        | CPU (numpy) | GPU compute | GPU end-to-end | CPU/GPU | CPU/e2e |
|----------|-------------|-------------|----------------|---------|---------|
| 10^5     | 0.13 ms     | 0.67 ms     | 0.37 ms        | 0.2x    | 0.4x    |
| 10^6     | 2.04 ms     | 0.32 ms     | 1.34 ms        | 6.4x    | 1.5x    |
| 10^7     | 21.2 ms     | 2.64 ms     | 11.3 ms        | 8.0x    | 1.9x    |
| 10^8     | 217 ms      | 26.8 ms     | 165 ms         | 8.1x    | 1.3x    |
| 4x10^8   | 919 ms      | 102 ms      | 1155 ms        | 9.0x    | 0.8x    |

Device write bandwidth (fill kernel): 19-23 GiB/s.

### Observations

- Pure compute speedup is a stable 6-9x for n >= 10^6; below 10^5 the
  launch/transfer overhead makes the CPU the winner.
- End-to-end (host<->device copies included) stays ahead 1.3-1.9x in
  the 10^6-10^8 sweet spot and falls behind at 4x10^8 - the device is
  transfer-bound (~4.5 GB/s copies vs 19-23 GiB/s write bandwidth;
  roughly three bytes moved per byte of useful work).
- Platform profile: strong compute, moderate memory bandwidth
  (shared/PCIe memory).  GPU executors are best used as resident batch
  kernels (data uploaded once, computed several times, fetched once);
  per-iteration round trips cancel the advantage at large n.
