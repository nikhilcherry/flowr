# flowr

A minimal, local-first pipeline runner. Stages are plain Python functions;
composing them lazily defines a DAG; every stage result is cached by a
content hash of **(code + params + inputs)**. Re-running an unchanged
pipeline is all cache hits; changing one parameter re-executes only the
affected downstream stages.

Pure Python, **stdlib only** — no config files, no daemons, no dependencies.
(numpy is optional: if present, array results are stored as inspectable
`.npz` files instead of pickles.)

```
pip install git+https://github.com/nikhilcherry/flowr
```

## Quickstart

```python
import flowr

@flowr.stage
def detrend(path, window=0.5):
    return load_and_flatten(path, window)      # any picklable return value

@flowr.stage
def search(lc, min_period=0.5):
    return find_peaks(lc, min_period)

lc    = detrend(flowr.File("data/tic123.npz")) # lazy — nothing runs yet
peaks = search(lc, min_period=0.8)

result = flowr.run(peaks)                      # executes; caches every stage
result = flowr.run(peaks)                      # instant: 100% cache hits

lcs   = flowr.map(detrend, [flowr.File(p) for p in paths])  # one node per item
peaks = flowr.map(search, lcs)
flowr.run(peaks, workers=4)                    # independent branches in parallel
```

The cache lives in `./.flowr/` (override with `FLOWR_DIR`). Change
`min_period` and only `search` re-runs; edit `detrend`'s logic and both
re-run; add a comment or docstring and **nothing** re-runs — code is hashed
after AST normalization, so formatting, comments and docstrings don't count.

## Seeing why things run: dry-run and `flowr why`

```python
plan = flowr.run(peaks, dry=True)   # executes nothing
print(plan)
# flowr plan: 42 nodes — 40 HIT, 2 MISS
#   HIT   detrend × 20
#   HIT   search × 20
#   MISS  aggregate  91c07d21e0f3  — param top_k: 5 -> 3
#   MISS  figure     ?             — upstream miss
```

Every miss carries a reason: `new node`, `code changed`,
`param min_period: 0.5 -> 0.8`, `file content changed: data/tic123.npz`,
`stage version: 1 -> 2`, or `upstream miss`.

The read-only CLI inspects the store from any directory:

```
flowr status                  # per-stage cache counts, store size
flowr runs                    # recent runs with hit/miss counts
flowr why 91c07d              # full provenance chain (git-style unique prefix):
                              #   stage, code hash, params, parents, run, duration
flowr gc --older-than 7d      # delete objects unreachable from recent runs
```

Exit codes: `0` ok, `1` usage/error, `3` nothing matched.

## How the cache key works

```
node_key = sha256(code_hash + stage_version + param_hash + upstream_result_hashes)
```

- **code_hash** — sha256 of the stage's AST-normalized source (docstrings,
  comments, whitespace and the decorator line stripped).
- **param_hash** — canonical serialization of the *bound* call arguments,
  defaults applied, so `f(x, window=0.5)` and `f(x)` hash identically when
  `0.5` is the default. Floats hash by `float.hex()`; dicts are
  order-insensitive; `flowr.File` hashes by streamed sha256 of the file's
  *content* (not its path). Any argument type flowr doesn't recognize is a
  hard error naming the argument and stage — it never silently `str()`-hashes
  an object.
- **upstream_result_hashes** — the hashes of the parent nodes' *stored
  results*, not their keys. This gives **early cutoff**: if you rewrite a
  stage but it produces byte-identical output, everything downstream still
  hits the cache. Two different code paths that compute the same bytes share
  their downstream cache. This is a feature.

## Honest limitations

### No closure tracing

flowr hashes the stage function's own source — **not** the helper functions
or modules it calls. If a stage calls `mylib.normalize()` and you edit
`normalize`, flowr will not notice. Two mandatory escape hatches:

```python
@flowr.stage(version="2")                     # bump to force invalidation
@flowr.stage(code_deps=[mylib, normalize])    # fold helpers' source into the hash
```

Tools that promise automatic transitive code hashing are lying to you or
importing half of PyPI. flowr makes the boundary explicit instead.

### Undeclared file reads are invisible

Only `flowr.File(path)` arguments participate in invalidation. A path passed
as a plain string is an opaque parameter: the stage can read the file, but
edits to it will **not** invalidate the cache — the same contract Make has.
Declare your inputs.

### Stages must be picklable, importable, top-level functions

Lambdas, closures, and REPL-defined functions are rejected with a clear
error. Results must be picklable (or numpy arrays). Stage functions receive
plain values: a `flowr.File` argument arrives as its path string; upstream
`Node`s arrive as their computed values.

## Determinism contract

flowr always guarantees cache **correctness**: a hit returns exactly the
bytes the original execution stored. It guarantees byte-identical **replay**
iff your stages are deterministic. Seeds are ordinary parameters — they're
in the cache key like everything else; there is no special seed machinery.
Workers set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
(BLAS threading reorders float reductions); set `FLOWR_KEEP_THREADS=1` to
keep your thread settings and give up bitwise replay. Avoid returning `set`s
(their pickle order varies across processes).

## Failure semantics

A stage retries per its `@flowr.stage(retries=N)` setting (default 0). A
node that still fails blocks its entire downstream subtree, **independent
branches keep running to completion**, and `flowr.run` then raises one error
summarizing every failed node with its traceback. Pass `fail_fast=True` to
abort at the first failure. Everything that finished stays committed — the
next run resumes from the frontier, and a run killed at any instant (power
loss included) leaves only fully-valid cache entries.

## Why not Snakemake / DVC / Make?

Those are fine tools with a different shape: rule files, YAML, CLI-first
workflows, and (for DVC) a hard git coupling. flowr is the useful 10% as a
pure-Python library — your pipeline is ordinary code, the DAG comes from
function composition rather than a config language, there is zero setup
beyond one `pip install git+…`, and the whole thing is stdlib-only so it
can't rot your environment. If you need cluster scheduling, containers, or
remote storage, use the big tools.

## Non-goals for v1

Remote/cloud storage, containers, cluster execution, scheduling daemons,
dynamic graphs (stages emitting stages), cache eviction policies beyond
`gc`, lambdas/closures as stages, cross-machine cache sharing.
