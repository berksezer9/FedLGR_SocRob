# Modifications for OfficeDB adaptation

This vendor repo (`nchuramani/FedLGR_SocRob`, GPL-3.0) is used elsewhere in this benchmark
as-is, per the root `CLAUDE.md`'s general "don't touch vendor" rule. This particular repo is
an exception: the FL/FCL OfficeDB experiments (`docs/fedlgr_officedb_design.md`) needed
changes vendor's shipped code structurally cannot support (exactly-2-task FCL, 8-class head,
750-stamp capacity ceiling, etc.) — the exception was raised with and approved by the user.
This file is the "let me know you changed it" record, per that agreement.

Branch: `officedb-adapt` (this nested git repo, independent of the main repo's history).
`main`/`origin/main` are left untouched as a diffable upstream reference — `git diff
main...officedb-adapt` shows the complete change set.

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
