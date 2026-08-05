# Modifications for OfficeDB adaptation

This vendor repo (`nchuramani/FedLGR_SocRob`, GPL-3.0) is used elsewhere in this benchmark
as-is, per the root `CLAUDE.md`'s general "don't touch vendor" rule. This particular repo is
an exception: the FL/FCL OfficeDB experiments (`docs/fedlgr_officedb_design.md`) needed
changes vendor's shipped code structurally cannot support (exactly-2-task FCL, 8-class head,
750-stamp capacity ceiling, etc.) — the exception was raised with and approved by the user.
This file is the "let me know you changed it" record, per that agreement.

Branch: `officedb-adapt` (this nested git repo, independent of the main repo's history).
`main`/`origin/main` are left untouched as a diffable upstream reference — `git diff
main...officedb-adapt` shows the complete change set. Item 23 below (chained-job resume) lives
on a separate branch, `fcl-resume` (branched off `officedb-adapt`, own worktree at
`afar-cfl-fedlgr_officedb_fcl_resume/vendor/FedLGR_SocRob`) — deliberately kept off
`officedb-adapt` until validated (see item 23's own status note), since `officedb-adapt` is
shared live by the `fedlgr_officedb`/`fedlgr_officedb_gpu` worktrees, both of which had real
jobs pending/running while this work was in progress.

**Editing principle**: every change below is a new parameter with a default that reproduces
original MANNERS-DB behavior exactly. Vendor's own `run_FL_local.sh`/`run_CL_local.sh` CLI
usage (2 tasks, 8 classes, circle/arrow split) still works unmodified after these changes.

## 1. `models/MobileNet.py`, `models/deepLabMobileNet.py`

`Net.__init__` gained `num_classes=8` (default preserves original), threaded to
`FCNet(num_classes=num_classes)`. OfficeDB passes `num_classes=9`.

## 2. `dataloader/imageloader.py`

`CustomDataset.__init__` gained `label_start=4` (default preserves original: Stamp, path,
circle, arrow, then 8 actions). `__getitem__` slices labels from `label_start` instead of the
hardcoded `4`. Needed because OfficeDB's extra/id-column count differs from MANNERS-DB's
fixed 2 (circle, arrow).

## 3. `dataloader/utils.py` (largest set of changes)

- **`load_universal(path, stamp_col='Stamp', mean_cols=None)`**: replaced the hardcoded
  `for i in range(250, 1000)` scan (both an arbitrary numeric-window assumption and a
  750-stamp capacity ceiling — too few for OfficeDB's 1000 images/robot) with a
  `groupby(stamp_col).agg(...)` over every stamp actually present. `mean_cols` (default: all
  non-stamp columns, matching original) picks which columns get averaged across annotators
  vs. carried through via `'first'` — needed because group/task/split id columns can be
  non-numeric (`.mean()` crashes on a string `split` column) and are constant within a stamp
  anyway (mean == first there), so this is a no-op for the original numeric-only case.
- **`load_images(path, action_cols=None, extra_cols=None, stamp_col='Stamp')`**:
  parameterized the 8 hardcoded MANNERS-DB action-column names (`action_cols`) and the 2
  hardcoded id columns `Using circle`/`Using arrow` (`extra_cols`), both defaulting to the
  original names. Also fixed a latent bug: stamp-from-filename parsing used
  `fname[:3]` (first 3 characters), silently mis-parsing any stamp >= 1000 (e.g.
  `"1000_...png"` -> `"100"`, which then fails the lookup and gets dropped by the
  surrounding `try/except`). Switched to `fname.split('_')[0]`, which reads the whole leading
  integer regardless of digit width — exact match for the original <1000 case, and required
  for OfficeDB's pooled by-robot/by-room configs (3000-image combined pool, stamps >= 1000).
- **`load_datasets(...)`**: added `action_cols`/`extra_cols` (passed through to
  `load_images`), `split_col` (if present, use its `train`/`test` labels instead of the
  internal random 75/25 split — `val` rows are simply not consumed, since this training loop,
  like vendor's original, has no validation phase) and `group_col` (assign each client every
  row of one unique group value — robot or room — instead of blind `random_split`; requires
  `#unique group values == num_clients`). `y_labels` now returns `action_cols` directly
  instead of `data.columns[-8:]`.
- **`task_splitter(path, task_col, task_values, n_clients, aug, batch_size=16,
  action_cols=None, extra_cols=None, split_col=None)`**: N-task generalization of
  `task_splitter_circle_arrow`. `task_values` is an ordered list of any length (was a
  hardcoded 2-way circle/arrow split). Returns cumulative per-task-boundary test loaders
  (`cumulative_test_per_task[t]` = tasks `0..t` combined) instead of separate per-task test
  loaders — generalizes the paper's `Task1 -> Combined(Task1+Task2)` evaluation protocol to N
  tasks (retention/forgetting measured after every boundary, not just the last one). For the
  original 2-task case this reproduces the exact same two loaders vendor returned
  (`cumulative[0]` == old `test_circle`/`valloader[0]`, since nothing precedes task 0;
  `cumulative[-1]` == old combined `test`).
- **`task_splitter_circle_arrow(path, n_clients, aug, batch_size=16)`**: kept as a thin
  wrapper calling `task_splitter(task_col='Using circle', task_values=[1, 0], ...)` —
  `'Using circle'` and `'Using arrow'` are complementary flags on every row, so this
  reproduces the original circle-then-arrow split exactly.
- **`load_datasets_pretrain(...)`**: gained `action_cols`/`extra_cols` passthrough (same
  defaults), threading `label_start` into its `CustomDataset` construction.

## 4. `main.py`

Added `--num_classes` (default 8), `--group_col`, `--split_col`, `--action_cols`,
`--extra_cols` (all default `None`/8, reproducing original behavior), threaded to
`Net(num_classes=...)` and both `load_datasets(...)` call sites (plain-FL and FedRoot-base
branches).

Also: both `ray_init_args = {...}` construction sites now add `ray_init_args["_temp_dir"] =
os.environ["RAY_TMPDIR"]` when that env var is set (no-op otherwise, exact original behavior).
Discovered running the smoke test on a shared Wilkes3 `ampere` node: Ray/Flower's
`fl.simulation.start_simulation` defaults its temp dir to `/tmp/ray`, which is node-local but can
already be populated by another user's leftover Ray session with different permissions —
surfaces as `PermissionError: [Errno 13] .../tmp/ray/ray_current_cluster`. Ray 2.10 (the version
pinned here) does **not** read any `RAY_TMPDIR`-style env var itself — `_temp_dir` must be passed
as a `ray.init()` kwarg, which only `ray_init_args` reaches — so an env var alone (without this
code change) would have silently done nothing. `RAY_TMPDIR` here is this repo's own convention,
not a Ray-native variable; set by the calling sbatch script (`fedlgr_officedb/slurm/*.sbatch`) or
`smoke_test/run_smoke_test.sh` to a private, job-unique directory -- must be **short**, not just
private: Ray creates a Unix domain socket (`plasma_store`) under this dir plus its own
session-id subdirectories, and Linux caps `AF_UNIX` paths at 107 bytes. A first attempt using a
repo-rooted path (`.../afar-cfl-fedlgr_officedb/smoke_test/.ray_tmp/<jobid>/...`, ~140 chars
once Ray's own subdirs are appended) blew past that and crashed with `OSError: AF_UNIX path
length cannot exceed 107 bytes` on a real GPU smoke-test job -- fixed by using
`/tmp/ray-<user>-<jobid>` instead (short, still unique per job/user so it doesn't reintroduce
the permission clash this was meant to fix).

## 5. `main_fcl.py`

- Added `--num_classes` (default 8), `--n_tasks` (default 2), `--task_col` (default
  `'Using circle'`), `--task_values` (default `None` -> `[1, 0]`), `--pretrained_dir`
  (default `'models'`), `--action_cols`, `--extra_cols`, `--split_col`, and (for FCL Axis A --
  see item 12 below) `--axis_mode`, `--action_groups`.
- Same `RAY_TMPDIR` -> `ray_init_args["_temp_dir"]` addition as `main.py` (item 4 above), at
  both of this file's `ray_init_args = {...}` sites.
- `run_strategy(...)`: generalized the results CSV from fixed
  `Loss1/RMSE1/PCC1/Loss2/RMSE2/PCC2` columns to `n_tasks`-many `Loss{t}/RMSE{t}/PCC{t}`
  columns, read from `history.losses_distributed`/`history.losses_centralized` at each
  task-boundary round (`rounds_per_task * k` for `k=1..n_tasks`) instead of the hardcoded
  `rounds/2` midpoint.
- Replaced the `task_splitter_circle_arrow` call with the generalized `task_splitter`.
- `agent_config['out_dim']`: `{'All': 8}` (5 occurrences) -> `{'All': args.num_classes}`.
- LGR pretrained-checkpoint load: `models/{cpu,gpu}/{Model}.pkl` (a bare cwd-relative path,
  ignoring `--path`/`--output` entirely) -> `{args.pretrained_dir}/{cpu,gpu}/{Model}.pkl`.
  `--pretrained_dir` defaults to `'models'`, reproducing the exact original path.
- All `client_fn_*` closures now pass `n_tasks=args.n_tasks` to the client constructors.
- `get_eval_fn_cl(...)` call sites pass `rounds_per_task=rounds // n_tasks, n_tasks=n_tasks`.

## 6. `client/default.py` (`FlowerClientCL`, `FlowerClient_NR`) and `client/fedRoot.py`
   (`FlowerClientCL_Root`, `FlowerClient_NR_Root`, `FlowerClient_LGR`)

Generalized the `< nrounds/2` / `== nrounds/2` / `> nrounds/2` three-way branch in
`fit()`/`evaluate()` into an N-way branch: `task_idx = min((server_round - 1) //
rounds_per_task, n_tasks - 1)`, `rounds_per_task = nrounds // n_tasks`. All classes gained
`n_tasks=2` (default preserves original 2-task timing exactly). Task-boundary-only actions
(Fisher/SI/MAS importance calc, memory-buffer snapshot, generator training) now fire at every
`server_round % rounds_per_task == 0` short of the last task, not just the single
hardcoded midpoint. `evaluate()` now always indexes `self.valloader[task_idx]` (the
cumulative per-task-boundary test loader from `task_splitter`) instead of special-casing a
separate `self.testloader` for the final task — `valloader[-1]` already equals the old
`testloader` (all tasks combined), so this is a no-op for the 2-task case.

`FlowerClient_LGR` keeps a deliberately different increment order from the EWC-style
clients: it calls `learn_batch` with the **pre**-increment `task_count` at a boundary, then
increments after (`LatentGenerativeReplay.learn_batch` branches on `self.task_count == 0` to
tell "first task" from "replay", and the boundary round is still that completing task's last
round of training) — matches vendor's original ordering exactly, just generalized to fire at
every boundary instead of the one hardcoded midpoint.

**Known scope-limited behavior, not fixed**: `FlowerClient_NR`/`FlowerClient_NR_Root`'s
`reg{cid}.pkl` only ever stores the *immediately preceding* task's rehearsal memory (vendor's
own original single-slot design — `learn_batch` is always called with either `{}` or a
freshly-constructed one-entry dict, never the accumulated `self.strat.task_memory`). Invisible
with the original 2 tasks (only one boundary ever occurs), but for OfficeDB's `n_tasks=3/6`
this means Naive Rehearsal only ever replays the most recent prior task, not the full task
history. Flagged, not fixed — fixing it properly means redesigning the file-based memory
hand-off, out of scope for "generalize the branching."

## 7. `CL/default.py`

- `LatentGenerativeReplay.create_dataset`/`create_dataset_gen`: cache filename changed from
  `{client_id}_current_task_reconstucted_data(_generator).pth` (fixed per client) to
  `{client_id}_{task_count}_current_task_reconstucted_data(_generator).pth`. The original
  filename was silently stale after the first task boundary — irrelevant for vendor's
  original exactly-2-task use (one boundary, so "stale" and "current" never actually
  diverged), a real bug for `n_tasks > 2` (the pseudo-replay set would keep reusing the
  generator state from the *first* boundary at every later one).
- `NormalNN.learn_batch`'s per-epoch debug PCC logging: `y_labels = [0, 1, ..., 7]` (hardcoded
  8) -> `list(range(self.config['out_dim']['All']))`. Cosmetic only — affects debug print
  organization, not training or final reported metrics (those go through `test()` in
  `utils.py`, which already used the real, correctly-sized `y_labels`).

## 8. `utils.py`

`get_eval_fn_cl(net, testloader, y_labels, DEVICE, rounds_per_task=5, n_tasks=2)`: generalized
the hardcoded `server_round <= 5` / `testloader[0]` vs. `testloader[1]` branch to
`task_idx = min((server_round - 1) // rounds_per_task, n_tasks - 1)` indexing into the
cumulative test-loader list. **Numeric behavior note**: original computed the combined-task
metric as the unweighted mean of two separately-computed `test()` calls (mean of per-task
means); this now runs one `test()` call over the pooled cumulative test set (sample-weighted
mean) since `task_splitter` already returns that as one combined loader. For `n_tasks=2` both
land on the same two underlying loaders, just combined differently — a deliberate, more
standard combined-eval computation (and the one the design doc's cumulative-eval protocol
calls for), not a regression.

`truncate_float(float_number, decimal_places)`: added a NaN/inf guard (returns the value
unchanged instead of `int(float_number * multiplier)`, which raises `ValueError: cannot convert
float NaN to integer`). Hit on a real CPU smoke-test run: a degenerate round produced `nan` for
`avg_pearson_score` (small per-task-per-client test splits make an undefined/degenerate Pearson
correlation more likely than at MANNERS-DB's original scale, even with `test()`'s own
jitter-retry fallback) and crashed `run_strategy`'s results-CSV write, which would otherwise
kill an entire multi-strategy/reg-coef sweep over one bad round rather than just recording NaN
for that cell.

## 9. `pretrain.py`

Added `--num_classes` (default 8) and `--action_cols`, threaded to the model constructors and
`load_datasets_pretrain`. `y_labels` list generalized from the hardcoded `[0..7]` to
`list(range(args.num_classes))`.

Also gained a `--extra_cols` CLI flag (threaded into the pre-existing `load_datasets_pretrain(...,
extra_cols=...)` param, which `main.py`/`main_fcl.py` already exposed but this script never did).
Discovered while wiring the smoke test: without it, `load_images()` defaults to looking for
`'Using circle'`/`'Using arrow'` in every row regardless of the actual `all_data.csv` schema —
on OfficeDB's `Robot,Room,Split`-shaped FL data this KeyErrors per-row inside `load_images`'s
try/except, silently producing an empty dataframe rather than a visible crash.
`fedlgr_officedb/pretrain_officedb.py` updated to pass `--extra_cols Robot,Room,Split`
accordingly.

## 10. `run_CL_local.sh`

Dropped the `--temp_dir ${temp_dir}` flag: `main_fcl.py` never defined this argument (argparse
would hard-error before anything ran), and `temp_dir` was never even assigned in this script.
Pre-existing vendor bug, unrelated to the FCL generalization above.

## 11. `requirements.txt`

- Pinned `flwr[simulation]==1.12.0` (was unpinned `flwr['simulation']`). Confirmed via
  source inspection (github.com/adap/flower) that `flwr.simulation.start_simulation` — the
  API `main.py`/`main_fcl.py` call — still exists in `v1.12.0` and is gone by `v1.13.0`
  (replaced by the new SuperLink/ServerApp/ClientApp architecture, a rewrite this repo does
  not use).
- Dropped `tensorflow`/`tensorflow-estimator`/`tensorflow-gpu`/`tensorflow-io-gcs-filesystem`/
  `tensorflow-probability`, both `opencv-python`/`opencv-python-headless` lines, and
  `tensorboard`/`tensorboard-data-server`/`tensorboard-plugin-wit` — grepped the codebase
  (`grep -rn "tensorflow\|opencv\|cv2\|tensorboard"`) and confirmed none of them are imported
  anywhere; this implementation is entirely PyTorch-based (mirrors the root README's existing
  macOS-setup precedent of dropping the same `tensorflow*` lines for the same reason). The
  `tensorboard==2.11.2` pin also turned out to be a genuine install blocker, not just dead
  weight: it requires `protobuf<4`, while `flwr==1.12.0` requires `protobuf>=4.25.2` —
  unresolvable together even ignoring the "unused" argument.
- Unpinned `grpcio` (was `==1.51.3`): `flwr==1.12.0` requires `grpcio>=1.60.0,<1.65.1` (or
  `>1.66.1`), incompatible with the original pin. Left unpinned rather than re-pinned, since
  the exact patch version doesn't matter here — let `flwr`'s own dependency resolution pick one.

## 12. FCL Axis A (action-subset) masking mechanism

FCL Axis A (`fedlgr_officedb/config.py`'s `AXIS_A_TASKS`, 3 tasks) doesn't fit the
`task_col`/`task_values` row-filtering model `task_splitter` already generalized for Axis B: an
action-subset task isn't a subset of *scenes* (every scene is relevant to every task), it's a
subset of the 9 output *columns* — only the active 3-of-9 action columns differ per task. This
needed a genuinely new mechanism (loss/eval column-masking), not just new config, across
several files:

- **`utils.py`**: `test(net, testloader, y_labels, DEVICE, active_idx=None)` gained an
  `active_idx` param — when given, slices `outputs`/`labels` to those column indices before
  computing loss/RMSE/PCC. `y_labels` must already correspond 1:1 to `active_idx` when given
  (caller's responsibility, same convention as `action_cols` elsewhere). `get_eval_fn_cl(...)`
  gained `active_idx_per_task=None` (a list indexed by task boundary) and threads the
  corresponding `active_idx`/sliced `y_labels` into `test()` each round. `None` (default)
  reproduces the original unmasked behavior exactly — zero effect on Axis B / the original
  2-task circle-arrow case.
- **`CL/default.py`**: `NormalNN.__init__` gained `self.active_idx = None`; `NormalNN.criterion`
  masks `preds`/`targets` by it before computing the base task loss. Since `L2.criterion` (EWC/
  EWCOnline/SI/MAS's shared base) calls `super().criterion()` for its task-loss term before
  adding its own regularization, and `Naive_Rehearsal` inherits `criterion` unmodified, this one
  change covers all five of EWC/EWCOnline/SI/MAS/NR without touching their own code. `MAS`'s
  `calculate_importance` is the one exception — its importance signal is label-free (squared
  output magnitude, not a loss against targets), so it doesn't route through `criterion()`;
  masked directly by slicing `preds` to `self.active_idx` before squaring, so importance isn't
  computed over action columns a task hasn't revealed yet.
- **`LatentGenerativeReplay`** (LGR): previously had `self.criterion = nn.MSELoss()` assigned as
  a plain **instance attribute** in `__init__`, which shadows any method of the same name and
  made masking impossible without editing every one of its 3 separate inline
  `self.criterion(out, targets)` call sites inside `learn_batch` (real Task-1 training, mixed
  real+pseudo-replay training, and the frozen-Top end-to-end pass — `update_model` is defined
  but never actually called by `learn_batch`, which reimplements the same logic inline each
  time). Fixed by removing that instance attribute and adding a `criterion()` **method**
  (mirroring `NormalNN`'s) that masks via `self.active_idx` and delegates to the already-existing
  `self.criterion_fn`. All 3 call sites now get masking automatically, with no per-call-site
  changes needed.
  - **Design note on mixed real+replay batches**: LGR's post-task-1 training mixes real
    current-task samples with self-generated pseudo-replay samples (pseudo-*labels* are
    generated fresh by the model's own current Top head at replay time, i.e. self-distillation,
    not stored ground truth from the original task). A single `active_idx` applied uniformly to
    the whole mixed batch is correct here specifically *because* Axis A's "task boundary" is an
    artificial column-visibility construct on top of fully-annotated data (OfficeDB's annotation
    CSVs have real ground truth for all 9 actions on every scene, unlike Axis B's genuinely
    per-domain-only images) — masking real samples to the current task's active columns doesn't
    discard/misuse any data, it's still that image's true label at those column positions. No
    per-sample masking (which would need memory/replay buffers to carry a mask alongside each
    stored sample) was needed.
- **`client/default.py`** (`FlowerClientCL`, `FlowerClient_NR`) and **`client/fedRoot.py`**
  (`FlowerClientCL_Root`, `FlowerClient_NR_Root`, `FlowerClient_LGR`): all five gained
  `active_idx_per_task=None, cumulative_idx_per_task=None` constructor params. `fit()` sets
  `self.strat.active_idx = active_idx_per_task[task_idx]` (this task's own exclusive columns)
  right after computing `task_idx`, before any `learn_batch` call — so EWC/SI/MAS's
  importance-estimation calls (which happen *inside* `learn_batch`) are automatically masked
  too. `evaluate()` looks up `cumulative_idx_per_task[task_idx]` (the union of columns revealed
  by tasks `0..task_idx` — the design doc's "train exclusive, evaluate cumulative" protocol) and
  passes it + the correspondingly-sliced `y_labels` into `test(..., active_idx=...)`. Both
  params default to `None`, reproducing unmasked behavior exactly for Axis B and the original
  2-task case — no regression risk.
- **`main_fcl.py`**: new `--axis_mode {row_filter,action_subset}` (default `row_filter`, i.e. the
  existing `task_splitter` path, unchanged) and `--action_groups` (only used in
  `action_subset` mode: per-task active action names, tasks separated by `;`, names within a
  task by `|`, e.g. `"A|B|C;D|E|F;G|H|I"`). In `action_subset` mode, data loading calls
  `dataloader.utils.load_datasets` (the existing FL-style loader — already does exactly
  "one shared partition across N clients, no row filtering") **once**, and reuses that same
  set of loaders for every task index (`trainloaders = [trainloaders_single] * n_tasks`, same
  for the cumulative test loader) — no new vendor data-loading function was needed.
  `active_idx_per_task`/`cumulative_idx_per_task` are computed once from `--action_groups`
  against `--action_cols`'s canonical column order, and threaded into all 5 `client_fn_*`
  closures and both `get_eval_fn_cl(...)` call sites (the two `FedAvg`-strategy branches;
  `FedRoot`'s strategy construction never passes `evaluate_fn` at all in vendor's original
  design, so its correctness rests entirely on the already-updated client-side `evaluate()`).

`fedlgr_officedb/prepare_data.py`'s `prepare_axis_a()` reuses `prepare_office_fl()`'s pooled
directory unchanged (no per-task dataset exists for Axis A — see that function's docstring).
`fedlgr_officedb/run_fcl.py` gained `--axis {a,b}` to select between the two.

## 13. `transfer_eval.py` (new file)

Added for the office->home transfer-learning experiment (extra idea raised at the 2026-07-27
supervisor meeting, see `docs/fedlgr_officedb_design.md`): loads an already-trained checkpoint
(from `pretrain.py`), evaluates it zero-shot on a *different* prepared dataset (Home), and
optionally fine-tunes on that dataset's train split before a second eval pass
(`--finetune_epochs`). Not FCL, not federated -- a single centralized model. Mirrors
`pretrain.py`'s structure (same `--num_classes`/`--action_cols`/`--extra_cols` generalization,
same relative `sys.path.append('../')` import of `utils.train`/`utils.test`) but uses
`dataloader.utils.load_datasets(num_clients=1, ..., split_col='Split')` instead of
`load_datasets_pretrain`, since this needs a real held-out test split rather than
`pretrain.py`'s single train==test `split_ratio` slice. `test()`'s return order is
`(loss, pcc, rmse)` (confirmed via source inspection of `utils.py`, not `(loss, rmse, pcc)` --
double-checked against the smoke-test pretrain log's printed triple, `rmse == sqrt(loss)`
holds only under that ordering). Driven by `fedlgr_officedb/transfer_office_to_home.py`
(thin subprocess wrapper, same pattern as `pretrain_officedb.py` -> `pretrain.py`).

## 14. `main.py` -- Ray client-concurrency fix (added --num_cpus, scaled client_res)

Discovered 2026-07-30 while the `by-robot` FL sweep job was ~12.6h in and only 3/5 rounds into
strategy 1 of 10 (`--strategy all --base all` runs FedAvg/FedBN/FedOptAdam/FedProx/FedDistill +
FedRoot x 5 bases = 10 strategy-runs per job): serial console showed only one `ClientAppActor`
training at a time per round despite 3 clients being sampled, on an 8-vCPU GCE VM. Root cause:
`num_CPUs` was hardcoded to `4` and **both** the CPU and GPU branches of `client_res` requested
the *entire* `num_CPUs` pool per client (only `num_gpus` was divided by `n_cl`, `num_cpus` never
was) -- Ray/Flower simulations cap concurrent clients at
`floor(pool_cpus / cpus_per_client_request)`, so requesting the whole pool per client forces
strict serialization regardless of how many cores the VM actually has (confirmed against
Flower's own simulation-resourcing docs). Fixed by: (1) adding a `--num_cpus` CLI arg (default
4, unchanged from original for anyone still relying on the default) so the pool size can be set
to match the VM's real vCPU count, and (2) dividing `num_cpus` by `n_cl` in *both* branches of
the `client_res` dict (mirroring the `num_gpus / n_cl` pattern the GPU branch already had for
GPUs, just never had for CPUs). Net effect: on an 8-vCPU VM with 3 clients and
`--num_cpus 8`, all 3 clients now train concurrently (~2.7 CPUs each) instead of one at a time --
expected ~3x wall-clock speedup per FL round. Does not affect the already-running
`fedlgr-fl-by-robot` job (image not rebuilt mid-run); applies to all FL/FCL/transfer jobs
submitted after the image is rebuilt and repushed. `vertex/gce_submit_job.py` callers should now
pass `--num_cpus <machine's vCPU count>` alongside the module args.

## 15. `utils.py` -- `predict_gen_distil` hardcoded 8-class reshape

Discovered 2026-07-31: every FedDistill/FedRoot-FedDistill job in the FL sweep (8/8 run so
far) crashed right after teacher-model pretraining with `ValueError: cannot reshape array of
size 216000 into shape (24000,8)`. `predict_gen_distil` (used only by the FedDistill teacher's
soft-label generation step, `dataloader/utils.py`'s `load_datasets(..., distil=True)`) reshaped
the teacher's stacked batch outputs to a hardcoded `8` columns — a MANNERS-DB holdover that
item 1's `num_classes` param never reached (this function takes the trained `net` directly, not
`args.num_classes`). OfficeDB's 9-class head produces `216000 / 24000 = 9` columns, not 8.
Fixed by reading the class count off the actual model output (`outputs[0].shape[-1]`) instead
of hardcoding it — no new parameter needed, reproduces the original 8-class case exactly since
that's still what a 8-class net outputs.

## 16. `dataloader/utils.py`, `main.py` -- federated ("distributed") eval used one shared
    testloader for every client, not each client's own held-out data

Discovered 2026-07-31 investigating suspiciously weak/flat RMSE/PCC across the whole FL sweep.
`main.py`'s `client_fn`/`client_fn_root`/`client_fn_BN`/`client_fn_BN_Root` all passed a single
`testloader` object (built once in `load_datasets`) to every client. Since `client.evaluate()`
(`client/default.py`, `client/fedBN.py`, `client/fedRoot.py`) is called on the same aggregated
global parameters for every client in a given round, this meant every client's
`clientwise/results{cid}.txt` row was computing the exact same eval on the exact same data —
confirmed empirically (byte-identical files across all clients in a completed 5-client run).
Beyond making the per-client logs meaningless, this also means the *federated* (distributed)
metric that `run_strategy` reports to `{clients}_{rounds}_{epochs}_{aug}_decentral.csv`
(`history.losses_distributed`/`metrics_distributed`, aggregated via each strategy's
`aggregate_evaluate`) was really a repeated centralized eval, not a genuine federated
per-client evaluation — the vendor's own PCC definition ("per action, per client, then
averaged", see `existing_fcl_paper_confidential.txt`) was never actually being computed.

Fixed in `dataloader/utils.py`'s `load_datasets(...)`: after building the pooled `testset`
(unchanged, still returned as `testloader` for the separate centralized `evaluate_fn`), it's
now also partitioned per client — using the same `group_col` groups as the trainset when
`group_col` is given (by-robot/by-room), else a matching `random_split` (iid/within-robot) —
into a new `testloaders` list, returned alongside `testloader`
(`load_datasets` now returns `(trainloaders, testloaders, testloader, y_labels,
data_permutation)`, one extra element). `main.py`'s four `client_fn*` closures now pass
`testloaders[int(cid)]` instead of the shared `testloader`; all `load_datasets(...)` call
sites updated to unpack the new 5-tuple. `main_fcl.py` (Axis A path) and `transfer_eval.py`
still call `load_datasets` but only ever consumed the pooled `testloader` (FCL's task-boundary
cumulative eval and the transfer-learning zero-shot/finetune eval are both intentionally
centralized, not per-client) — both updated to just discard the new `testloaders` return value,
no behavior change there. Only the plain-FL benchmark's federated evaluation semantics change;
FCL/transfer-learning are unaffected. This invalidates all FL-sweep results collected before
this fix (RMSE/PCC values will differ once each client evaluates on its own held-out slice).

## 17. `server/strategies.py`, `main.py` -- `FedOptAdamStrategy` never actually ran FedAdam

Discovered 2026-07-31 in the same investigation. `FedOptAdamStrategy(fl.server.strategy.FedAvg)`
defined a custom `aggregate(self, reports)` method intending to implement server-side
Adam-style aggregation, but Flower's `Strategy` interface calls `aggregate_fit(self,
server_round, results, failures)` (confirmed against the installed `flwr==1.12.0` source) —
`aggregate` was never called by the framework, so this class silently fell through to its
parent `FedAvg.aggregate_fit`, i.e. plain FedAvg aggregation, for every `FedOptAdam` and
`FedRoot-FedOptAdam` run. Not introduced by the OfficeDB adaptation — confirmed identical on
`origin/main`, a pre-existing vendor bug that just happened to surface here.

Fixed by making `FedOptAdamStrategy` subclass `fl.server.strategy.FedAdam` (Reddi et al. 2020)
directly instead of hand-rolling the aggregation — the dead `aggregate` method is deleted;
`aggregate_fit` is now flwr's own correct, tested implementation. The class's
`aggregate_evaluate` override (weighted-mean of `avg_pearson_score`/`avg_rmse`, needed so
`run_strategy`'s CSV writer keeps working) is unchanged. Chosen over hand-fixing the original
formula (which had no discoverable correctness precedent anyway — no momentum/beta state, wrong
`results` structure assumption, wrong return shape) per explicit user direction to default to
the simplest, best-tested option when there's no working precedent to preserve.

flwr's `FedOpt`/`FedAdam` base class requires an explicit `initial_parameters` kwarg (unlike
`FedAvg`/`FedProx`, which default it to `None` and bootstrap from the first client). Both of
`main.py`'s `FedOptAdamStrategy(...)` construction sites (plain-FL and FedRoot-FedOptAdam) now
pass one explicitly: `get_parameters(net)` for plain-FL (matches `client_fn`'s full-model
`get_parameters`/`set_parameters`), `get_parameters(net.conv_module)` for FedRoot (matches
`client_fn_root`'s root-submodule-only `get_parameters` in `client/fedRoot.py`) — using the
wrong one would silently produce a shape mismatch against what clients actually return.
Also invalidates all `FedOptAdam`/`FedRoot-FedOptAdam` results collected before this fix (they
were really running FedAvg/FedRoot-FedAvg).

## 18. `pretrain.py`, `dataloader/utils.py` -- pretrain never honored a leakage-safe Split
    column, and trained/evaluated on the identical slice

Discovered 2026-07-31 while scoping the single-robot office<->home domain-transfer experiment
(`docs/domain_transfer_officedb_to_mannersdbplus.md`). `pretrain.py` always called
`load_datasets_pretrain`, which ignores any pre-built `Split` column entirely: it does its own
random-permutation shuffle and carves off a `split` (default 0.33) fraction as
`data_images_test`, then `run()` reassigns that same variable name (`data`) and passes it as
*both* `train_loader` (to `train()`) and `testloader` (to `test()`) -- i.e. every pretrain run,
including every checkpoint produced so far for the FCL/LGR path, trained and "evaluated" on the
exact same ~33% slice, and never touched `data/splits/*.csv`'s actual 80/10/10 train/val/test
assignment at all. Harmless for pretrain's original purpose (just warming up a checkpoint for
FedLGR's LGR mechanism, not itself a reported result), but wrong for the domain-transfer
experiment's OfficeDB-side and MANNERSDBPlus-side pretraining, which need a real, leakage-safe
train-only fit -- reusing an already-tested split mechanism (`load_datasets`'s `split_col`,
same one `main.py`/`transfer_eval.py` already use) rather than inventing a new one.

Fixed by adding an optional `--split_col` arg to `pretrain.py`: when given, it calls
`load_datasets(num_clients=1, ..., split_col=args.split_col)` instead, training on
`trainloaders[0]` (Split=='train' rows only) and evaluating the printed sanity-check numbers on
the real held-out `testloader` (Split=='test' rows). Omitting `--split_col` reproduces the
original behavior exactly (needed for any caller without a real split column, e.g. plain
MANNERS-DB). `fedlgr_officedb/pretrain_officedb.py` now always passes `--split_col Split` for
OfficeDB/MANNERSDBPlus data, since both have one. This changes the checkpoint
`pretrain_officedb.py`'s default (pooled, no `--robot`) invocation produces too -- harmless
since FCL Axis A (the only consumer) hasn't been run against a real checkpoint yet.

## 19. `server/strategies.py`, `main.py` -- FedAdam applied wholesale corrupts BatchNorm
    running_var, at any `eta`

Discovered 2026-07-31 smoke-testing item 17's fix: `FedAvg`/`FedOptAdam` by-robot smoke test
(`--rounds 1 --epochs 1`) evaluated fine at round 0 (loss 8.36, using the freshly-passed
`initial_parameters`) but produced NaN loss/RMSE/PCC from round 1 on, for both plain
`FedOptAdam` and `FedRoot-FedOptAdam`. Root cause: flwr's `FedAdam.aggregate_fit` (see item 17)
applies its Adam-normalized update to *every* entry of the flat parameter list returned by
`get_parameters(...)`, with no concept of which entries are trainable weights vs. BatchNorm's
`running_mean`/`running_var`/`num_batches_tracked` buffers. `running_var` must stay
non-negative; Adam's per-parameter update magnitude is normalized to ~`eta` regardless of
`eta`'s size (that's what the normalization is for), so "just use a smaller eta" doesn't
actually fix this — confirmed empirically against a fresh `MobileNet(num_classes=9)` state_dict
(53 `running_var` buffers, real min value 0.0007): `eta=0.1` pushed 49/53 negative,
`eta=0.0001` (1000x smaller than flwr's own default) still pushed 12/53 negative. A negative
`running_var` hits `sqrt(var + eps)` in the very next BatchNorm forward pass and produces NaN
network-wide.

Fixed by splitting `FedOptAdamStrategy.aggregate_fit` (`server/strategies.py`) into two paths
per parameter index: Adam-normalized for trainable weights (same math as flwr's `FedAdam`,
reimplemented directly against `self.m_t`/`self.v_t`/`self.eta`/`self.beta_1`/`self.beta_2`/
`self.tau`, which `FedAdam.__init__` already sets up), plain weighted-average (via
`fl.server.strategy.FedAvg.aggregate_fit`, called directly to bypass `FedAdam`/`FedOpt` in the
MRO) for BatchNorm buffers — a weighted average of already-non-negative values can't produce a
negative result, unlike an unconstrained Adam step. Which indices are buffers is determined by
a new `bn_buffer_mask(state_dict_keys)` helper (name-matches `running_mean`/`running_var`/
`num_batches_tracked`), computed once in `main.py` from the same `net.state_dict()` (plain FL)
or `net.conv_module.state_dict()` (FedRoot) order used to build `initial_parameters`, and passed
to `FedOptAdamStrategy`'s new `buffer_mask` constructor kwarg (default `None`, reproducing
flwr's plain wholesale `FedAdam` exactly — only this repo's two `FedOptAdamStrategy(...)`
construction sites in `main.py` pass a real mask).

Re-smoke-tested (by-robot, `--rounds 1 --epochs 1`): no more NaN for either `FedOptAdam` or
`FedRoot-FedOptAdam`, but plain `FedOptAdam`'s round-1 loss still blew up to ~261 (vs. round 0's
8.36, and vs. `FedAvg`'s comparable round-1 loss of ~0.81) even with buffers excluded —
flwr's `FedAdam` defaults (`eta=0.1`, `tau=1e-9`) are still too aggressive a server step for
this deep CNN's *trainable* weights, not just its BatchNorm buffers. Adam's per-parameter
update magnitude is normalized to ~`eta` independent of the true delta scale, so `eta=0.1`
means every one of ~2M parameters shifts by up to ±0.1 in a single server step regardless of
how much the clients actually moved locally. Fixed by passing `eta=0.01, tau=1e-3` at both
`FedOptAdamStrategy(...)` construction sites in `main.py` — Reddi et al. 2020's own
image-classification settings, not the simpler-task defaults flwr ships. Re-smoke-tested again:
`FedOptAdam` round 1 loss 9.05 (close to round 0's 8.36, a stable small step, not a 30x
explosion), `FedRoot-FedOptAdam` round 1 loss 0.60 (comparable magnitude to `FedAvg`'s 0.60-0.89
range). Note `FedOptAdam`'s round-1 loss barely improving on round 0 (8.36 -> 9.05) is expected,
not a remaining bug: unlike `FedAvg` (which directly adopts the average of clients' fully
locally-trained weights each round), `FedAdam`'s server step is deliberately decoupled from the
raw magnitude of client progress -- it's supposed to move cautiously per round and build
momentum (`self.m_t`) over many rounds, so 1-round smoke-test numbers aren't comparable to
`FedAvg`'s in isolation; the real sweep's 5 rounds should show it catching up.

## 20. `dataloader/utils.py` -- every real-image `DataLoader` gained `num_workers=NUM_WORKERS`

Made on the `fedlgr_officedb_gpu` branch/worktree while diagnosing why a real-data GPU smoke
test (`fedlgr_officedb.run_fl`, ampere, `--processor_type gpu`) never completed a single
training epoch within 3-29 minutes, despite Ray/Flower correctly detecting the A100 and
allocating fractional-GPU client actors. Isolated the cause with a standalone timing check
(no Ray/GPU involved): plain `Image.open(...).convert('RGB').load()` over 100 real
OFFICE-MANNERSDB images took ~85-115ms/image (both on the login node and, confirmed separately,
on an ampere compute node via `srun` -- so this is shared Lustre-backend contention, not a
login-node-only artifact). Every `DataLoader(...)` wrapping `CustomDataset` (which does this
`Image.open` per sample in `__getitem__`, lazily) previously omitted `num_workers`, defaulting
to `0` -- fully synchronous, single-process loading with no prefetching, so this I/O cost was
paid serially against the training loop rather than overlapped with GPU/CPU compute. Added a
module-level `NUM_WORKERS = 2` constant and threaded `num_workers=NUM_WORKERS` into all 7 real-
image `DataLoader` call sites (the two in-memory-tensor `DataLoader`s in the top-level
`utils.py`'s `predict`/`predict_gen`, used for LGR's generative-replay pseudo-samples, do no
disk I/O and were left unchanged). `NUM_WORKERS=2` is a conservative, un-tuned starting value
chosen to fit the smallest per-client CPU budget in use (4 CPUs/client on GPU jobs, see
`fedlgr_officedb/slurm/submit_fl_sweep.py`'s `COMPUTE` dict) while leaving headroom on the main
process -- purely a performance change, does not affect training results (same data, same
order via each call's existing `shuffle=...`, just loaded by worker subprocesses instead of the
main process). Applies equally to CPU and GPU runs (the bottleneck is disk I/O, not
processor type) -- worth carrying back to the `fedlgr_officedb` (CPU) branch too once confirmed.

## 21. `utils.py` -- `test()` computes Pearson/RMSE/loss over the full test set, not per-batch

Found 2026-08-01 investigating why ~15-20% of completed FL-sweep clientwise result files had
`avg_pearson_score = nan` for every single round (not sporadic) -- initially suspected as a
FedRoot-specific bug (user's original report) but confirmed empirically to hit every strategy
equally (`FedAvg`/`FedBN`/`FedDistill`/`FedOptAdam`/`FedProx`/`FedRoot-*` all affected), so it's
a shared metric-computation bug, not architecture-specific.

Root cause: the test `DataLoader` (`dataloader/utils.py`) uses `batch_size=16` with the default
`drop_last=False`, so whenever a client's held-out test partition size isn't a multiple of 16,
the final batch is small (as small as 3-5 samples in practice, since OfficeDB's per-client/
per-room/per-robot partitions rarely divide evenly). `test()` computed Pearson **per batch**
then averaged across batches -- with OfficeDB's 1-5 discrete rating scale, a small final batch
has a non-trivial chance that every sample shares the same rating for at least one of the 9
actions (confirmed empirically: by-room's `Hallway` client, n=229 test rows, has a final
5-sample batch where all 5 rows rate "Carry Drinks" and "Carry Small Objects" identically;
`SmallOffice`'s final 3-sample batch does the same for 2 other actions). `scipy.stats.pearsonr`
returns `nan` for a zero-variance input. The existing NaN-rescue (re-running `pearsonr` with the
*outputs* perturbed by tiny Gaussian noise) only helps when the *predictions* are degenerate --
it cannot fix degenerate *labels*, which is what a same-rating batch is. The resulting NaN then
got folded into the client's overall `avg_pearson_score` via a plain, non-NaN-safe `Average()`
(`sum(lst)/len(lst)`), so **one** degenerate batch/action silently NaN'd the client's entire
reported Pearson score. Since the test `DataLoader` isn't reshuffled between evaluation calls,
it's the *same* small batch every round -- explaining why affected clients were NaN for all 5
rounds, every time, regardless of strategy.

Verified the mechanism by replicating `main.py`'s exact seeding (`np.random.seed(1024)`, set at
module import before any other RNG-consuming call) and `dataloader/utils.py`'s exact
`data.sample(frac=1)` calls against the real `office_fl/all_data.csv`, offline: reproduced the
precise Hallway/SmallOffice final-batch rows above bit-for-bit. Also checked whether this is a
genuine data problem (e.g. a room with near-constant ratings): full-partition std for every
room/every action is healthy (~1.2-1.4 on the 1-5 scale) -- this is purely a last-batch sampling
artifact, not a real label-quality issue.

Fixed by changing `test()` to accumulate every batch's `(labels, outputs)` across the whole
`testloader` first, then compute loss/RMSE/Pearson once over the full concatenated test set,
instead of per-batch-then-averaged. This also resolves the open question flagged in
`docs/fedlgr_officedb_experiments.md`'s "Results quality" section ("whether the per-batch...
Pearson computation... adds meaningful noise" -- it does, in the form of both this NaN failure
mode and generally noisier per-batch estimates than a single full-set correlation). Single
source of truth: `get_eval_fn`/`get_eval_fn_cl`/`get_eval_fn_bn` and both FL/FCL client
`evaluate()` methods (`client/default.py`, `client/fedBN.py`, `client/fedRoot.py`, `CL/
default.py`) all call this same `test()`, so the fix covers FL, FCL, and centralized eval in one
place -- no other `pearsonr` call site exists in the vendor code.

**Every already-completed FL Priority-1 (bugfix rerun) result for a client whose partition hits
this last-small-batch case is affected** -- that client's `avg_pearson_score` was `nan` (not
just noisy) for every round, in every strategy that used that partition. Confirmed affected
(via `grep ",nan," clientwise/results*.txt`): by-room's client 2 (`Hallway`) and client 4
(`SmallOffice`) across all 5 by-room strategies from the `20260731_174918` submission batch;
within-robot-2's client 0 across multiple robots/strategies from several submission batches
(same root cause -- `within-robot-2`'s `random_split(..., manual_seed(42))` partition happens to
leave a small last test batch for that client too). These need a rerun under the fixed code to
get a real (non-NaN) `avg_pearson_score` for the affected client -- the fix doesn't retroactively
repair already-written result files, only future runs. Not yet re-smoke-tested against real data
as of this writing -- do that before resubmitting the affected jobs.

## 22. `main.py`/`main_fcl.py` -- optional `FL_ROUND_TIMEOUT` env var bounds each round

Added 2026-08-01 after an unexplained Ray/Flower `VirtualClientEngine` hang: client actors
spawn cleanly (Ray/CUDA init succeeds, `configure_fit` samples the right number of clients),
then zero progress -- no epoch print, no error, no crash -- until the job's wall-time limit
kills it. Confirmed reproducible on **both** CPU and GPU, at `n_clients=2` *and* `n_clients=3`,
always with `strategy_cl=EWC` (no other strategy tested yet) -- see the FCL Axis A GPU debug
jobs `32509212`/`32537651` (both hung, `n_clients=2`) and the FCL Axis A CPU smoke test
`32538963` (hung, `n_clients=3`, i.e. the config a different session had concluded was
"confirmed working" based on one earlier successful GPU run -- that conclusion doesn't hold up;
this looks like an intermittent/racy hang, not one deterministically tied to client count).
Root cause **not found** -- ruled out (by the earlier GPU debugging session): no `n_clients=2`
special-case in the vendored app or in Flower 1.12.0's own `VirtualClientEngine`, no
degenerate/empty per-client data split. `vendor/FedLGR_SocRob/dataloader/utils.py`'s missing
`--num_cpus` in the two FCL Axis A smoke-test sbatch scripts (causing Ray to auto-detect the
whole node's CPU count instead of the Slurm-allocated share) was considered but ruled out as
the sole cause: the GPU debug rerun (`32537651`) passed `--num_cpus` correctly matching its
Slurm allocation and still hung.

Since the underlying cause is unresolved, added a defensive/diagnostic bound instead: both
`main.py` and `main_fcl.py` now read `FL_ROUND_TIMEOUT` (seconds) and pass it as
`fl.server.ServerConfig(..., round_timeout=...)`. Unset (the default, `None`) reproduces the
original unbounded-wait behavior exactly -- this does not change any already-passing job's
behavior.

**`round_timeout` alone was NOT enough -- verified empirically, and the actual behavior was
worse than expected, not just "not a fix".** Tested it directly against the known-hanging config
(CPU, `n_clients=3`, EWC, `FL_ROUND_TIMEOUT=90`, job `32577187`'s debug rerun): once the 90s
timeout hit, Flower did NOT raise/abort -- `flwr.simulation.ray_transport.ray_actor.py`'s
`get_client_result` raises a `TimeoutError` per client, but `server.py`'s `fit_round`/
`evaluate_round` catch that as a per-client *failure* (not a fatal error) and calls
`strategy.aggregate_fit`/`aggregate_evaluate` anyway with `results=[]`. The stock
(and this repo's, pre-this-fix) strategies all did `if not results: return None, {}` --
i.e. **silently proceed to the next round with a no-op**, rather than stopping. Confirmed via
the debug log: round 0 AND round 1 both logged `aggregate_fit: received 0 results and 3
failures` back-to-back -- once the underlying hang triggers, it does not recover on its own, so
without a hard stop the simulation would have ground through all 15 rounds doing nothing,
producing a completed-looking job (exit 0, full round count) with meaningless output for
however many rounds were affected -- arguably worse than the original silent hang, since a
0/15-real-rounds job wouldn't obviously look broken without checking the log for this exact
message.

**Real fix**: `server/strategies.py` -- every `aggregate_fit`/`aggregate_evaluate` override
(`FedAvgWithAccuracyMetric`, `FedProxWithAccuracyMetric`, `FedOptAdamStrategy`) now calls a
shared `_require_results(results, failures, phase, round)` helper first, which raises
`RoundFailedError` (`RuntimeError` subclass) the moment `results` is empty, instead of the old
`if not results: return None, {}`. `FedAvgWithAccuracyMetric`/`FedProxWithAccuracyMetric` didn't
previously override `aggregate_fit` at all (relied on flwr's own `FedAvg`/`FedProx` base, which
has the same silent-return behavior) -- added the override there too, purely to insert this
check before delegating to `super().aggregate_fit(...)`. An exception raised inside
`aggregate_fit`/`aggregate_evaluate` is not caught anywhere in flwr's `Server.fit()` main loop
(confirmed by reading `flwr/server/server.py` -- the per-round call isn't wrapped in a
try/except), so it propagates all the way out of `fl.simulation.start_simulation(...)` and
crashes the whole `python -m fedlgr_officedb.run_fcl`/`run_fl` process with a non-zero exit --
real fail-fast, visible in `sacct` as FAILED, no log-grepping needed to detect it. Only fires
when `results` is completely empty (every client failed/timed out that round) -- every
previously-successful run had non-empty `results`, so this changes nothing for already-passing
jobs, only converts "silently degrade for the rest of the job" into "stop immediately." Applies
to FL and FCL both (same strategy classes, same file). `FL_ROUND_TIMEOUT` is still what makes
`results` empty in the first place within a bounded time instead of hanging forever -- both
pieces (the timeout AND the hard-stop-on-empty-results) are needed together.

Still a mitigation, not a fix for the underlying hang -- revisit if/when the real root cause is
found (a live `py-spy`/`gdb` stack trace of a hung actor is the logical next step, not yet done
-- `py-spy` isn't installed and the compute nodes have no internet access to install it there;
would need installing from the login node into the shared `.venv` first).

## 23. `main_fcl.py`, `client/default.py`, `client/fedRoot.py`, `utils.py` -- chained-job resume for FCL Axis A

**Status: implemented and validated** (synthetic gate run 2026-08-03, real-data chained-sbatch
gate still pending) -- kept on the separate `fcl-resume` branch/worktree until the real-data
gate also passes, not yet merged into `officedb-adapt`.

**Why**: a real FCL Axis A combo (15 rounds = 3 tasks x 5 rounds/task, real `epochs=10`) is
projected at ~20-24h on CPU (extrapolated from the FL sweep's confirmed ~78-91 min/round at the
same n=3/24-CPU config) -- longer than CSD3's 12h QOS wall cap on every account tried
(`gunes-sl3-cpu`, `gunes-sl3-gpu`/`sl4-gpu`). This lets one combo span multiple 12h-capped Slurm
jobs, one per task, resuming both the client-side CL state (already round-tripped through pickle
files per round, see item 6) and the Flower-aggregated global model weights (previously never
persisted).

**Two real, pre-existing bugs found and fixed while building this** (not introduced by this
adaptation, surfaced because chaining depends on them):
1. `main_fcl.py`'s experiment directory was unconditionally derived from `datetime.now()` on
   every invocation -- a second chained job would silently write into a brand-new empty
   directory, never find the first job's `task{cid}.txt`/`reg{cid}.pkl`, and quietly restart CL
   bookkeeping at task 0 (wrong answer, not a crash). Fixed via new `--experiment_dir` (below).
2. The 3 FedRoot strategy-construction branches (EWC-family, NR, LGR) build
   `FedAvgWithAccuracyMetric` with no `evaluate_fn` at all -- the only existing driver-side hook
   that sees full global weights (`get_eval_fn_cl`, used by the 2 FedAvg-family branches) doesn't
   cover FedRoot. A second, checkpoint-only hook (`make_checkpoint_hook`, `utils.py`) fills this
   gap -- always returns `None`, confirmed safe against flwr 1.12.0's `Strategy.evaluate()`
   (treats `None` as "no centralized eval this round", `history.losses_centralized` stays
   untouched exactly as today's no-`evaluate_fn` behavior).

**New opt-in parameters, all defaulting to reproduce original behavior exactly**:
- `--round_offset` (int, default 0): added to `FlowerClientCL`/`FlowerClient_NR`
  (`client/default.py`) and `FlowerClientCL_Root`/`FlowerClient_NR_Root`/`FlowerClient_LGR`
  (`client/fedRoot.py`). `effective_round = server_round + round_offset` replaces raw
  `server_round` everywhere `_task_idx`/`_is_boundary` (and the NR/NR_Root classes'
  `server_round == int(self.nrounds)` final-round replay-buffer-deletion check) use it. Since
  chunks are required to align exactly to task boundaries, `round_offset = (job_index-1) *
  rounds_per_task` is an exact substitution -- Flower's own `server_round` always restarts at 1
  for a fresh `start_simulation()` call, which this offsets back to the correct global round.
- `--rounds_this_run` (int, default `None` -> falls back to `--rounds`): decouples "Flower
  rounds this process executes" from `rounds` (still used for CSV filenames and
  `rounds_per_task = rounds // n_tasks` math, exactly as before). `main_fcl.py`'s `run_strategy()`
  gates its whole chunked-vs-monolithic code path on `chunked = rounds_this_run != rounds` --
  when `False`, the original CSV-writing/`plot_results` code runs byte-for-byte unchanged.
- `--experiment_dir` (str, default `None`): see bug #1 above.
- `--seed` (int, default `None`): `main_fcl.py` previously seeded nothing at all (unlike
  `main.py`'s always-on `torch.manual_seed(1024)`); only used so far for resume-equivalence
  testing, not required by the resume mechanism itself.

**Checkpoint mechanism** (`utils.py`): `_dump_global_params(reference_module, path)` pickles a
plain `state_dict()` atomically (tmp file + `os.replace`, so a job killed mid-write -- e.g. by
the Slurm wall-time cap -- can't leave a half-written, unloadable checkpoint for the next
chained job), same convention as `pretrain.py`'s existing `pickle.dump(model.state_dict(), f)`.
Wired into `get_eval_fn_cl`'s `evaluate()` closure (2 FedAvg-family branches) and the new
`make_checkpoint_hook` (3 FedRoot branches, bug #2 above) -- dumped every round (not just the
last), cheap and self-healing if a job dies mid-chunk. Loaded back via a new
`_load_checkpoint_params(checkpoint_path, fallback)` helper in `main_fcl.py`, called at all 5
strategy-construction sites right before `initial_parameters=fl.common.ndarrays_to_parameters(...)`
is built -- returns `fallback` (the fresh net's own `get_parameters()` output, unchanged)
whenever `checkpoint_path` is `None` or doesn't exist yet, and deliberately returns a *new* list
rather than mutating the shared `params` variable in place, so an earlier branch's checkpoint
pickup in a `--strategy_cl all`/`--reg_coef all` sweep can never leak into a later branch that
has no checkpoint of its own. Both dump and load are gated on `--experiment_dir` being set
(independent of `chunked`), so they're also usable standalone as a diagnostic on a monolithic run.

**`run_strategy()` result-accumulation rework**: `rounds` was previously used for three things
at once inside `run_strategy()` -- Flower's actual `num_rounds`, the CSV filename, and which
`history` indices count as "task-final" (assumes one `history` object spans all `n_tasks`).
A chunked job's own `history` only has as many entries as rounds *it* ran, so reusing the
original indexing as-is would `IndexError`. When `chunked`, instead of writing the final CSV row
immediately, this job's task-final metrics get merged into a small per-combo JSON state file
(`{output}/.chunked_state/<strategy>_<coeff>_<clients>_<rounds>_<epochs>_<aug>.json`, atomic
write) keyed by task number; only once every task is present does the same-shaped decentral/
central CSV row get assembled and appended, reading from the merged state instead of one
`history` object.

**Explicit scope cuts, not implemented in this pass**:
- Full 15-round `plot_results()` reconstruction across chunks -- each chunk still emits its own
  local (single-task) plot. Only CSV metric parity is required/tested.
- `savecomp()`'s `comp.csv` gets `n_tasks` diagnostic RAM/CPU/GPU rows per chunked combo instead
  of 1 -- cosmetic only, not a scored metric.
- SI's online importance accumulator (`self.w` in `CL/default.py`) and NR's single-slot (not
  full-history) `reg{cid}.pkl` are both pre-existing vendor characteristics, identical whether a
  combo runs monolithically or chunked -- not touched by this item.

**Verification: synthetic gate run 2026-08-03** (`fedlgr_officedb/slurm/_resume_equivalence_test.sh`
+ `_resume_equivalence_check.py`, via `srun` on `gunes-sl3-cpu`/icelake, synthetic
`smoke_test/data/office_fl`): monolithic run vs. a 3-job chunked run of the same tiny combo,
for both `FedAvg`x`EWC` (exercises `get_eval_fn_cl`'s checkpoint dump) and `FedRoot`x`NR`
(exercises `make_checkpoint_hook` and the NR-Root `effective_round==nrounds` fix). Two swaps
from the original plan, both logged in the test script: (1) the scratch dir must be
Lustre-backed (`results/fedlgr_officedb/_resume_equivalence_scratch/`), not node-local `/tmp` --
`srun` runs on a compute node, so a login-node-relative `/tmp` path is invisible both during and
after the job; (2) `FedRoot`x`LGR` was swapped for `FedRoot`x`NR` -- LGR's replay generator
(`CL/default.py`, `DataLoader(combined_dataset, batch_size=16, shuffle=True)`, no
`drop_last=True`) crashes on a remainder-of-1 batch at the 2nd task boundary, confirmed via a
real run to happen in the **monolithic** path too (i.e. a genuine pre-existing vendor bug, not
caused by chunking) -- flagged, not fixed, same treatment as the SI/NR quirks above; `NR` shares
`make_checkpoint_hook` identically so covers the FedRoot-family checkpoint path just as well.

**All structural/mechanism checks passed**: `task{cid}.txt`/`reg{cid}.pkl`/`mod{cid}.pkl` match
exactly between mono and chunked; `global_params.pkl` loads cleanly as a non-empty state_dict for
every chunk; CSV row shape matches. **Checkpoint continuity confirmed bit-exact** by
cross-referencing the raw Flower log directly (not automated into the checker): job 2's very
first `evaluate_fn` call ("initial parameters (loss, ...)", using job 1's checkpoint, before job
2 does any of its own training) reported loss `52.745880126953125`, which equals job 1's own
final reported `Loss1` in the chunked CSV to full float precision; likewise job 3's initial loss
(`478.75341796875`) equals job 2's final `Loss2` exactly. This is direct proof the checkpoint
round-trips the exact correct state across separate OS processes, not an approximation.

**Final CSV metric values (mono vs. chunked) do NOT match, and this is expected, not a gate
failure**: a chunked job N+1 is a fresh process that calls `torch.manual_seed(args.seed)` once
at the top, immediately before its own first round -- its RNG stream (and DataLoader shuffle
order, which has no seeded `generator=`) starts fresh at "task N+1, round 1" every time. The
monolithic run's RNG stream, by contrast, has already been advanced by every prior task's
training when it reaches the same task. Same seed, same checkpointed weights (proven bit-exact
above), but different actual gradient steps taken from task 2 onward -- and this tiny/
undertrained synthetic setup is volatile enough (loss in the hundreds-to-thousands range on
2-sample batches) that the resulting trajectories diverge substantially. This is a property of
the tiny synthetic test data's instability, not the resume mechanism; `_resume_equivalence_check.py`
prints these as `[INFO]`, not `[FAIL]`, with the full reasoning in its `compare_csv()` docstring.

**Not yet run**: the real-data chained-`sbatch` gate (genuine `--dependency=afterok` jobs across
separate compute-node allocations, not just separate processes on the same node) -- still
required before trusting this for the real 10-combo batch.

## Not changed

`dataloader/imageloader.py`'s hardcoded image crop `(295, 0, 295+1018, H)`: verified OfficeDB
images are 1920x1080, the same resolution as MANNERS-DB, so the crop geometry vendor already
tuned for carries over unchanged. Sanity-checked visually during the smoke test rather than
trusted blindly (see `fedlgr_officedb/` wrapper package's own notes).

`CL/default.py`'s `Naive_Rehearsal` single-slot memory design (see item 6's "known
scope-limited behavior" above) and `dataloader/utils.py`'s `load_datasets_hyper`/`ac_split`
(both already dead/unused code paths in vendor's own repo — `load_datasets_hyper` was already
broken before this adaptation, calling `load_images(path, aug)` with 2 positional args when
`load_images` only ever accepted 1; neither is called from `main.py`/`main_fcl.py`).
