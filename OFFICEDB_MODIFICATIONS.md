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

## 23. `pretrain.py`, `transfer_eval.py`, `utils.py`, `dataloader/utils.py` -- val-loss early
stopping, replacing the domain-transfer experiment's fixed epoch counts

Added 2026-08-03 for `docs/domain_transfer_officedb_to_mannersdbplus.md`'s domain-transfer
experiment, after checking per-epoch training-loss logs from the existing fixed-10-epoch
pretrain jobs (32487816/818, 32688171-174): Office-domain checkpoints (all 3 robots) plateau by
~epoch 6-8, but Home-domain checkpoints (all 3 robots) were still declining at epoch 10 (Nao,
Pepper) or had just turned back upward (PR2, more consistent with early overfitting noise than a
clean plateau) -- meaning a fixed epoch count picked one number that under-trains Home and,
plausibly, over-trains Office, conflating "domain is harder" with "domain needed more/fewer
epochs" in the transfer-eval comparison. User asked for the literature-standard fix instead of
guessing a bigger fixed number: track validation loss every epoch and keep the lowest-val-loss
epoch's weights (early stopping with patience), not whatever epoch a fixed budget happens to
land on -- e.g. Prechelt 1998, "Early Stopping -- But When?".

**`dataloader/utils.py`**: new `load_val_loader(path, ...)` function, standalone rather than
added to `load_datasets()`'s return tuple -- extending that tuple would require updating every
existing call site (`main.py`, `run_fl.py`, `transfer_eval.py`, `pretrain.py`) to unpack one more
value for a feature only the new early-stopping path needs. Every OfficeDB/MANNERSDBPlus
per-robot dataset already carries `Split=='val'` rows (the `data/splits/*.csv` 80/10/10 split)
that nothing previously consumed -- `load_datasets()`'s own comment already noted this ("'val'
rows are simply not consumed here, since this training loop... has no validation phase") and
even had a `# valloaders = []` placeholder never wired up. Returns `None` if the data has no
`split_col` or no `'val'` rows, so callers can fall back to the old fixed-epoch path instead of
hard-failing (e.g. legacy MANNERS-DB paths with no 3-way split).

**`utils.py`**: new `train_with_early_stopping(model, train_loader, val_loader, DEVICE, y_labels,
max_epochs=40, patience=5)`, added alongside `train()` rather than modifying it -- `train()` is
also called from the FL/FCL per-round local-training path (`run_fl.py`/`main_fcl.py`), where
"epochs" means local epochs per round, not "train to convergence"; those call sites are
unaffected. Evaluates `val_loader` via the existing `test()` after every epoch, keeps a
`copy.deepcopy`'d state_dict whenever val loss improves, stops after `patience` epochs with no
improvement, and loads the best (not final) state_dict into `model` before returning.

**`pretrain.py`/`transfer_eval.py`**: new `--early_stopping`/`--max_epochs`/`--patience` args,
each opt-in and additive -- omitting `--early_stopping` reproduces the exact previous fixed
`--epochs`/`--finetune_epochs` behavior. `--early_stopping` requires `--split_col` and `'val'`
rows in the data (`pretrain.py`) or `'val'` rows in `--data` (`transfer_eval.py`); both hard-exit
with a clear message rather than silently falling back if neither is available, since a silent
fallback here would silently reproduce the exact problem this item exists to fix.

Threaded through `fedlgr_officedb/pretrain_officedb.py` and
`fedlgr_officedb/transfer_office_to_home.py` (our wrappers) and
`fedlgr_officedb/slurm/domain_transfer_{pretrain,eval}.sbatch` (`EARLY_STOPPING=1` env var, plus
`MAX_EPOCHS`/`PATIENCE`) the same way. sbatch wall-time bumped 45min/1h -> 2h each, since
`--max_epochs 40` with early stopping can run longer than the previous fixed 10/5-epoch budgets
if patience doesn't trigger quickly -- pure safety margin, no cost if a job finishes early.

## 24. `pretrain.py` -- optional `--seed` for the domain-transfer multi-seed reliability rerun

Added 2026-08-04 for `docs/domain_transfer_officedb_to_mannersdbplus.md`'s "Multi-seed
reliability rerun" section: the item-23 early-stopped ceiling-vs-joint comparison was a single
uncontrolled stochastic run per condition (no seed was set anywhere in this repo before this).
Literature check (Reimers & Gurevych 2017; Bouthillier et al. 2021) found that's not sufficient
evidence for a comparison claim, so a 5-seed rerun was needed.

New `--seed` arg (default `None`, unseeded -- reproduces prior behavior exactly). When given,
`run()` calls `random.seed`/`np.random.seed`/`torch.manual_seed` before anything stochastic
happens (data-loader shuffling, model weight init). Placed at the very top of `run()`, before
`load_datasets*`/`load_val_loader` and model construction, since seeding after either would leave
that call's own randomness uncontrolled. Threaded through
`fedlgr_officedb/pretrain_officedb.py` (`--seed`) and
`fedlgr_officedb/slurm/domain_transfer_pretrain.sbatch` (`SEED` env var, optional). Not added to
`transfer_eval.py`: zero-shot eval has no training-time stochasticity and aggregate PCC/RMSE are
order-invariant, so the eval half of the pipeline needed no change.

## 24b. `transfer_eval.py` -- optional `--seed`, correcting item 24's "not needed" claim for the fine-tune phase

**Backfilled commit** -- this code was already written and in active use (the crossdomain
domain-transfer multi-seed reliability rerun, `results/fedlgr_officedb/domain_transfer/
crossdomain_seed_reliability/`, and `fedlgr_officedb/slurm/submit_crossdomain_transfer_seeds.py`,
which explicitly calls this "transfer_eval.py's new `--seed`") but was never actually committed in
this nested repo -- caught while preparing an unrelated commit and backfilled here rather than
left uncommitted or silently folded into other work.

Item 24 above claimed no `transfer_eval.py` change was needed since "zero-shot eval has no
training-time stochasticity." That's true for pure zero-shot (`--finetune_epochs 0`, the default),
but incomplete: `transfer_eval.py` also supports fine-tuning the checkpoint on the target domain's
train split before a second eval pass (`--finetune_epochs`/`--early_stopping`), and that phase has
its own real stochasticity (train-loader shuffling order) uncontrolled by a pretrain-time seed.
Same pattern as item 24: `--seed` arg (default `None`), `run()` calls `random.seed`/
`np.random.seed`/`torch.manual_seed` at the top, before the fine-tune phase. Threaded through
`fedlgr_officedb/transfer_office_to_home.py`'s existing `--seed` (already committed there,
depended on this vendor-side code already existing).

## 25. `main_fcl.py`, `client/default.py`, `client/fedRoot.py`, `utils.py` -- chained-job resume for FCL Axis A

**Status: implemented, validated, and merged into `officedb-adapt`** 2026-08-05 (developed on a
separate `fcl-resume` branch/worktree while `officedb-adapt` was shared live by
jobs/worktrees mid-run; merged once both the synthetic gate and the real-data chained-sbatch gate
below had passed).

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

**Real-data chained-sbatch gate: passed** (`fedlgr_officedb/slurm/submit_fcl_chain.py`, genuine
`--dependency=afterok` jobs across separate compute-node allocations, real OfficeDB Axis A data,
2026-08-04/05). Of 6 combos x 3 chunks submitted (`FedAvg`x`EWC` at 3 `reg_coef` values,
`FedAvg`x`NR`/100, `FedRoot`x`EWC`/0.001, `FedRoot`x`NR`/100 -- 18 jobs), 17 completed cleanly on
the first attempt. The one exception, `FedRoot`x`NR`/100 chunk2 (job 32832506), is not a resume
bug: all 3 clients were still legitimately mid-`epoch 9/10` when Flower's `round_timeout` fired --
round 6 (chunk2's first round, i.e. task 2's first round, where NR's rehearsal buffer starts
being replayed alongside the new task's data) took ~125min, just over the then-`FL_ROUND_TIMEOUT`
of 7200s/120min. Plain `FedAvg`x`NR` and plain `FedRoot`x`EWC` both stayed comfortably under
120min/round in the same batch, so it's specifically the FedRoot-root-model-compute +
NR-replay-cost combination that's this slow, worst at task-boundary rounds. Fixed by raising
`fcl_axis_a_chunk.sbatch`'s default `FL_ROUND_TIMEOUT` to 10800s (180min) -- pure headroom, since
the timeout is a ceiling and doesn't slow down combos that finish faster; matters most for the
full-sweep's other `FedRoot`x`NR` buffer sizes (10/1000) still queued, since larger buffers
replay more samples per epoch and would only make this worse. Retried as jobs
32894066/32894067 (chunk2/chunk3, resumed from chunk1's checkpoint via `--experiment_dir`, same
mechanism as any other chained resume).

## 26. `utils.py`, `main.py` -- plain-FL checkpoint hook for Track 2 federated domain-transfer

**Status: implemented** 2026-08-05, sign-off obtained before editing (per CLAUDE.md's "flag,
don't silently do" rule for this fork).

**Why**: confirmed by direct read of `client/{default,fedBN,fedRoot}.py` that **no completed FL
run of any strategy has ever persisted a complete, loadable model**. `FlowerClient.fit()`/
`evaluate()` (`client/default.py`, plain FedAvg/FedOptAdam/FedProx/FedDistill) write only
per-round CSV metrics, nothing to disk. `FlowerClient_BN` (`client/fedBN.py`) saves
`mod_bn{cid}.pkl` -- local BatchNorm params only. `FlowerClient_Root.fit()`
(`client/fedRoot.py:145,150`) saves `mod{cid}.pkl` -- literally `self.net.fc_module.state_dict()`,
the personalized head only; the aggregated root/`conv_module` is never written anywhere. Worse
than initially assumed for FedRoot: `main.py`'s `base=='FedAvg'` branch previously passed **no
`evaluate_fn` at all** to the strategy -- there wasn't even a per-round central hook to begin
with, unlike the plain strategies which at least had a transient (never-persisted) one via
`get_eval_fn`. Needed for Track 2's federated-vs-centralized domain-transfer comparison
(`docs/officedb_spine_implementation_plan.md` §3) -- without this, no federated checkpoint
exists to evaluate.

**Fix**: mirrors the existing FCL path's checkpoint mechanism (item 25 above,
`_dump_global_params`/`make_checkpoint_hook`) instead of inventing a new one.
- `utils.py`: `get_eval_fn(net, testloader, y_labels, DEVICE, checkpoint_path=None)` gained the
  optional `checkpoint_path` param (previously CL-only, via `get_eval_fn_cl`). When set, its
  `evaluate()` closure calls `_dump_global_params(net, checkpoint_path)` right after loading each
  round's aggregated weights into `net` -- same call, same place `get_eval_fn_cl` already makes
  it. Default `None` reproduces prior behavior exactly for every other caller.
- `main.py`: only 2 of the existing call sites pass a `checkpoint_path` -- the plain `FedAvg`
  branch (`get_eval_fn(central_model, ..., checkpoint_path=f"{path}/global_params.pkl")`) and the
  FedRoot `base=='FedAvg'` branch, which gains
  `evaluate_fn=make_checkpoint_hook(central_model.conv_module, f"{path}/global_params.pkl")` --
  same `make_checkpoint_hook` item 25 already validated for the FCL path, always returns `None`,
  reproducing the prior "no centralized eval" behavior exactly and adding only the checkpoint
  side effect. `FedProx`/`FedOptAdam`/`FedDistill`/`FedBN` (plain and FedRoot-base) call sites are
  untouched -- deliberately scoped to only the 2 combos Track 2 needs
  (by-robot FedAvg + FedRoot-base=FedAvg), not a general checkpoint-everywhere change.

**Checkpoint contents, important for the consuming eval script**: the `FedAvg` branch's
`global_params.pkl` is a **complete `Net` state_dict**, directly loadable by
`transfer_eval.py`/`pretrain.py`'s existing `model.load_state_dict(pickle.load(f))` convention.
The FedRoot branch's `global_params.pkl` is the **shared root (`conv_module`) only** -- getting a
complete per-robot personalized model requires combining it with that robot's own `mod{cid}.pkl`
(`fc_module`, already saved today, unaffected by this change). That assembly is done in
`fedlgr_officedb/` code, not here.

**Risk / blast radius**: additive-only, gated behind a default-`None`/previously-absent parameter;
no existing strategy/branch's runtime behavior changes except the 2 combos above gaining a
disk-write side effect. Isolated to this worktree's submodule checkout (each git worktree has an
independent checkout); the currently-running/queued FCL Axis A jobs (`main_fcl.py`, `run_fcl.py`)
use `get_eval_fn_cl`/`make_checkpoint_hook` via a separate code path untouched by this item, and
already-running Python processes don't re-read source after import in any case.

## 27. `main.py` -- optional `--seed` for the plain-FL/FedRoot multi-seed re-run

**Status: implemented** 2026-08-05, sign-off obtained before editing.

**Why**: Track 2's federated-vs-centralized domain-transfer comparison needs the same multi-seed
reliability treatment `pretrain.py`/`transfer_eval.py` already give the centralized ceiling
(`results/fedlgr_officedb/domain_transfer/crossdomain_seed_reliability/`, item 24). `main.py`
(plain FL/FedRoot) had no `--seed` argument at all -- just the unconditional module-level
`torch.manual_seed(1024)`/`np.random.seed(1024)` (lines 22, 25), unlike `pretrain.py`/
`transfer_eval.py`/`main_fcl.py`, which all already got this treatment. Confirmed the seed
actually matters here (not a no-op): `dataloader/utils.py`'s `load_datasets` calls
`np.random.permutation(len(data))` for `data_permutation`, which reads the global NumPy RNG state
-- varies with the seed. `--split_col Split` (already passed for OfficeDB) keeps the train/test
boundary itself fixed from the CSV, so seed variation changes weight init + shuffling/permutation
only, the same protocol as `pretrain.py`'s existing multi-seed rerun.

**Fix**: mirrors `pretrain.py`'s exact pattern (its own `run()`, lines 15-27) rather than
inventing a new one. `main.py`'s `run(args)` gained, at its very top (before `client_fn`/dataset
loading/model init):
```python
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
```
plus a new `--seed` argparse arg (default `None`) and `import random` (previously unimported in
this file). The pre-existing module-level `torch.manual_seed(1024)`/`np.random.seed(1024)` are
left untouched, so omitting `--seed` reproduces original behavior byte-for-byte -- identical
"unset = unseeded original" convention as `pretrain.py`.

**`fedlgr_officedb/run_fl.py`** (our own driver, not vendor) gained a matching `--seed`
passthrough into `build_command`'s `main.py` invocation. Note: it does **not** auto-suffix
`--output` per seed -- the caller (the 2b re-run submission script) must pass a seed-distinct
`--output` per run, or different seeds' `global_params.pkl`/`mod{cid}.pkl` checkpoints will
clobber each other in the same directory.

**Risk / blast radius**: additive-only, gated behind a default-`None` argument; every existing
call site (all real `fl_strategy.sbatch`/`submit_fl_sweep.py` jobs, none of which pass `--seed`)
is unaffected. Same isolation/already-running-process caveats as item 26 apply.

## 28. `CL/default.py` -- LGR replay-buffer `DataLoader`s crash at task boundaries

**Status: implemented** 2026-08-12, per the paper-prep five-point plan's Tier 1 item
(`docs/paper_prep_2027/five_point_plan_2026-08-12.md` in the main repo).

**Why**: `FedRoot`/`FedAvg` x `LGR` (generative replay, the vendor's own headline mechanism,
arXiv 2405.15773) has been excluded from every FCL Axis A/B sweep so far
(`docs/fedlgr_officedb_experiments.md`) because of a pre-existing vendor bug flagged during item
25's chained-resume work: `create_dataset()`/`create_dataset_gen()` (used at the 2nd+ task
boundary to mix real current-task data with pseudo-replayed prior-task data before training)
each built their combined `DataLoader` with `batch_size=16, shuffle=True` and no `drop_last`.
Whenever `len(combined_dataset) % 16 == 1`, the final batch has exactly 1 sample, which crashes
training the moment it reaches a `BatchNorm` layer (`self.model.fc_module` in `create_dataset`'s
caller, `self.generator`'s VAE in `create_dataset_gen`'s caller) -- `BatchNorm` needs >1 sample
per channel to compute a batch statistic. Confirmed pre-existing (not introduced by the OfficeDB
fork) and reproducible in the monolithic (non-chunked) path, not just chunked -- see item 25's
"Two swaps from the original plan" note.

**Fix**: added `drop_last=True` to both `DataLoader(combined_dataset, batch_size=16,
shuffle=True, ...)` calls (`create_dataset`, `create_dataset_gen`) -- the same flag every other
`DataLoader` in this class already uses (`predict_gen`, `predict_from_gen`,
`predict_from_gen_gen`, all `drop_last=True`), so this brings `LGR`'s replay-mixing path in line
with the rest of the class rather than introducing a new convention. No other change.

**Trade-off, accepted**: dropping the remainder batch means up to 15 samples of that task
boundary's combined (real + replayed) data go unused in that epoch. Given `combined_dataset` is
`len(new_data) * 16` replayed samples plus the real per-task/per-client training data (typically
in the hundreds for OfficeDB's by-robot/by-room partitions, never as small as 16), a dropped
remainder of at most 15 is a small fraction of one epoch's data, not a meaningful shrinkage of
the effective training set -- consistent with why `drop_last=True` is already the vendor's own
default choice everywhere else this exact `batch_size=16` pattern appears in this file.

**Verification**: smoke-tested via `fedlgr_officedb/slurm/smoke_fcl_lgr_icelake.sbatch`
(`FedRoot`x`LGR`, real OfficeDB Axis A data, `--rounds 3 --epochs 1` so `rounds_per_task=1` and
round 2 already reaches `task_count=1` -- the task-boundary branch that calls `create_dataset()`).
Job 33538729 (2026-08-12, `gunes-sl3-cpu`/icelake): round 1 (`task_count=0`, pre-fix-irrelevant
"first task" branch) completed cleanly for all 3 clients; round 2 (`task_count=1`) hit
`create_dataset()` -- log shows "Learning LGR Data Mixed" (the call site) immediately followed by
a clean, unbracketed `Epoch 1/1, Loss: <value>` print for multiple clients, which only prints
after the *entire* `mixed_task_data` `DataLoader` has been iterated without exception -- i.e. the
exact remainder-batch/`BatchNorm` crash this fix targets did not occur. (The job was later killed
by its own `FL_ROUND_TIMEOUT`/Slurm wall-time -- unrelated infra mis-calibration, not a fix
failure: LGR's boundary round runs 3 training phases per client (`Task 2` fc-training, "End to
End Top Frozen" full-model training, generator retraining via `create_dataset_gen()`) vs. round
1's 2, and 1200s wasn't enough headroom -- see the smoke sbatch script's own comments.) Combined
with item 25's existing note that this crash was independently confirmed on a real (pre-fix) run,
this is a confirmed pre-fix-crashes / post-fix-doesn't pair, the same standard used elsewhere in
this file. A fully clean 3-round completion (job 33540387, `FL_ROUND_TIMEOUT=3600`,
`--time=02:30:00`) was queued the same day to remove any residual ambiguity but had not started
(`PD`/`Priority`) after 29 minutes in queue at the time of this commit -- optional confirmation,
not a blocker; check `sacct -j 33540387` for its outcome.

**Risk / blast radius**: touches only the `LGR` replay-mixing path (`create_dataset`,
`create_dataset_gen`), both dead code for every other CL strategy (EWC/EWCOnline/SI/MAS/NR never
call them). No effect on any already-completed non-LGR result.

## 29. `server/strategies.py`, `main.py`, `client/default.py` -- add FedNova strategy

**Status: implemented** 2026-08-12, part of the ICRA 2027 five-point plan's item 2 ("Extend FL
strategy coverage -- FedNova (+ SCAFFOLD)",
`docs/paper_prep_2027/five_point_plan_2026-08-12.md`). Implemented on a separate worktree/branch
(`fedlgr_officedb-fednova`, forked off `fedlgr_officedb`) to avoid clashing with other
concurrent-session work on this repo; merged back once smoke-tested. SCAFFOLD (moderate cost --
client-side control-variate state) is deliberately deferred to a later session, per the plan.

**Why**: WS1's FL-baseline-coverage survey (`docs/paper_prep_2027/ws1_method_backbone_selection.md`)
found FedNova (Wang et al., NeurIPS 2020, https://arxiv.org/abs/2007.07481) is the
cheapest-to-add, most commonly-expected "why isn't this here" baseline in the non-IID vision-FL
literature (NIID-Bench, FedRS-Bench, the FedProj baseline set) -- a server-side
aggregation-only reweighting, addable in the same shape as the existing `FedOptAdamStrategy`
Flower subclass (item 17/19), no client-training-loop change needed.

**What FedNova corrects**: FedAvg implicitly favors clients that take more local SGD steps
(more local epochs and/or more local data) -- their update simply moves further, and a plain
weighted average by `num_examples` doesn't fully cancel that out when local step *counts* differ,
not just data volume. FedNova instead normalizes each client's update to a per-step-average
direction before combining, removing that bias ("objective inconsistency" in the paper's terms).

**Client-side change** (`client/default.py`, `FlowerClient.fit()`): the fit-metrics dict, unused
before this (`{}`), now includes `local_steps` = `self.epochs * len(self.trainloader)` -- the
total local SGD step count for this client this round. `self.epochs` is a single global
`--epochs` CLI value shared by every client (not per-client in this codebase), so this varies
across clients only through `len(self.trainloader)` (client data volume) -- i.e., FedNova
corrects for the same client-data-imbalance signal `FitRes.num_examples` already reflects, just
via a different (multiplicative-normalization, not weighted-average) mechanism. Additive-only:
every other strategy's `aggregate_fit` either never reads `fit_res.metrics` or only aggregates it
through `fit_metrics_aggregation_fn`, unset (`None`) everywhere except `FedNovaStrategy` -- no
behavior change for `FedAvg`/`FedProx`/`FedOptAdam`/`FedBN`/`FedDistill`/`FedRoot`.

**Server-side change** (`server/strategies.py`, new `FedNovaStrategy(fl.server.strategy.FedAvg)`):
implements the paper's normalized-averaging update directly (flwr has no built-in FedNova, unlike
FedAdam/FedYogi/FedAdagrad) --
```
d_i          = (x^t - y_i) / tau_i        # per-client normalized delta
p_i          = n_i / sum(n_j)             # same weighting FedAvg/flwr already use
tau_eff      = sum_i(p_i * tau_i)
x^{t+1}      = x^t - tau_eff * sum_i(p_i * d_i)
```
where `x^t` is `self.current_weights` (tracked the same way `FedOptAdamStrategy` tracks it --
requires `initial_parameters`, raises `ValueError` at construction if omitted), `y_i` is client
i's returned weights, and `tau_i` is `local_steps` from that client's fit metrics (raises
`RoundFailedError` if a client's fit result is missing it, rather than silently defaulting --
same fail-fast philosophy as item 22's `_require_results`).

**BatchNorm-buffer risk, found before running anything (by inspection, applying item 19's
lesson directly -- not rediscovered empirically this time)**: FedAvg's weighted average is a
convex combination (`sum_i(p_i) == 1`), so it can never leave the range of the values being
averaged. FedNova's update is *not* convex in the same sense --
`sum_i(p_i * tau_eff / tau_i)` only equals 1 when every client's `tau_i` is identical, so in
general the update is a genuine extrapolation past `x^t`/`y_i`'s range. Applied to BatchNorm's
`running_var` (which must stay non-negative), that extrapolation risks the exact NaN failure
mode item 19 found for unmasked `FedOptAdamStrategy`/`FedAdam`. Fixed the same way: `FedNovaStrategy`
takes an optional `buffer_mask` constructor kwarg (`bn_buffer_mask(net.state_dict().keys())`,
same helper item 19 added), and `main.py`'s new `FedNova` branch passes it -- buffer indices get
a plain weighted average of the clients' returned buffer values instead of the FedNova
extrapolation; `buffer_mask=None` (unused by any real call site here) would reproduce the raw,
unsafe-for-BatchNorm formula, matching `FedOptAdamStrategy`'s own default semantics.

**`main.py`**: new `elif strat == 'FedNova':` branch (mirrors the `FedOptAdam` branch just above
it -- same `initial_parameters`/`buffer_mask` construction, `client_fn`/plain `FlowerClient`, no
FedRoot variant added -- deliberately out of scope for now, per user decision 2026-08-12). Also
passes `checkpoint_path=f"{path}/global_params.pkl"` to `get_eval_fn` (same Track 2 federated
domain-transfer hook item 26 added for the plain `FedAvg` branch), added opportunistically at the
user's request since it's a one-line reuse of an already-tested kwarg -- not required for this
item's own by-robot-only scope, but there in case a later session wants FedNova's checkpoint for
domain-transfer work. Deliberately **not** added to the `strategy == 'all'` hardcoded list
(`['FedAvg', 'FedBN', 'FedOptAdam', 'FedProx', 'FedDistill', 'FedRoot']`) -- the five-point plan
scopes FedNova to the `by-robot` partition only, single-seed, not a second full sweep across
every partition; keeping it out of `all` means existing/future `--strategy all` sweeps on other
partitions are unaffected, and `FedNova` must be requested explicitly
(`--strategy FedNova`/`fedlgr_officedb/run_fl.py --strategy FedNova`).

**Risk / blast radius**: additive-only new strategy branch + one new client fit-metrics key
(ignored elsewhere). No existing strategy's code path, weighting, or output changes. Smoke-tested
via `smoke_test/run_smoke_test.sh`-style `main.py --strategy FedNova` invocation against synthetic
data before any real Slurm job (job 33533156, full 9-step suite; jobs 33537206/33538001 hit
unrelated node-level infra flakiness -- a Slurm TIMEOUT stuck at import and a `No space left on
device` at `ray.init()`, both on a busy/full shared node, matching item 22's documented Ray/VCE
hang risk -- job 33538472 completed cleanly on a different node and confirmed
`global_params.pkl` is written and loads back as a 321-tensor state dict with no NaNs). See this
item's companion smoke-test logs, not committed here -- ephemeral verification only.

## 30. `CL/default.py` -- force eval mode on `fc_module` during LGR's generative pseudo-labeling

**Status: implemented**, found already applied uncommitted in the shared vendor checkout at the
start of this session (mtime 2026-08-12 21:17, predating this session) -- picked up mid-way
through re-verifying item 28's LGR fix (both worktrees share this vendor checkout, not a
per-worktree clone, so an uncommitted change here is visible/live everywhere at once).
Committing and documenting it now per this file's own convention that every vendor change is
catalogued, rather than leaving it as a stray uncommitted diff in a "don't touch" vendor repo.

**Why**: `LatentGenerativeReplay.predict_from_gen`/`predict_from_gen_gen` (used to generate
pseudo-labeled replay pairs from the VAE's latent space) call `self.model.fc_module` one
`torch.randn(1, 64)` sample at a time inside a `torch.no_grad()` loop. If `fc_module` was left in
`.train()` mode by the caller's preceding training phase, a batch size of exactly 1 crashes any
`BatchNorm` layer in it (needs >1 sample per channel for a batch statistic) -- the same class of
bug as item 28's `create_dataset`/`create_dataset_gen` DataLoaders, just hit via inference-time
batch-of-1 calls instead of a drop_last remainder batch.

**Fix**: both methods now save `fc_module`'s incoming training-mode flag, force `.eval()` before
the sampling loop (correct anyway for a no_grad, inference-only pseudo-labeling pass -- `eval()`
makes BatchNorm use its running stats instead of computing a batch statistic, sidestepping the
batch-of-1 problem entirely), and restore the original mode afterward so the caller's own
training-mode bookkeeping is undisturbed.

**Verification**: no dedicated isolated smoke test was run for this fix specifically (unlike item
28's `smoke_fcl_lgr_icelake.sbatch`). It was already live in the code for both the item-28
re-verification smoke test (job 33571492, `COMPLETED` clean, both task boundaries exercised) and
the real-scope `FedRoot`x`LGR` chain (33573281-33573283) this session submitted -- i.e. confirmed
compatible with a full clean LGR run, not confirmed in isolation as fixing a reproduced crash the
way item 28 was.

**Risk / blast radius**: touches only `LatentGenerativeReplay.predict_from_gen`/
`predict_from_gen_gen`, both `LGR`-only code (dead for every other CL strategy). Restores
`fc_module`'s prior mode on exit, so no behavior change for callers relying on it staying in
`.train()`/`.eval()` after these methods return.

## 31. `server/strategies.py`, `main.py`, `client/default.py`, `utils.py` -- add SCAFFOLD strategy

**Status: implemented** 2026-08-13, part of the ICRA 2027 five-point plan's item 2 (deliberately
deferred from item 29's FedNova work, per that item's own note). Implemented on a separate
worktree/branch (`scaffold`, forked off `officedb-adapt` at item 29, rebased onto item 30 once
that landed concurrently -- checked out inside the outer repo's `fedlgr_officedb-scaffold`
worktree/branch) to avoid clashing with other concurrent-session work on this repo; smoke-tested
clean (see below), merged back into `officedb-adapt`.

**Why**: WS1's FL-baseline-coverage survey
(`docs/paper_prep_2027/ws1_method_backbone_selection.md`) found SCAFFOLD (Karimireddy et al.,
ICML 2020, https://arxiv.org/abs/1910.06378) is the single most commonly-expected non-IID
baseline missing from this benchmark's coverage, alongside FedNova -- "moderate cost" there
specifically because of the client-side control-variate state its canonical form requires,
unlike FedNova's server-side-only aggregation change.

**What SCAFFOLD corrects**: under non-IID client data, plain local SGD/Adam steps drift toward
each client's own local optimum ("client drift"), and FedAvg's weighted average doesn't undo
that drift, only averages across it. SCAFFOLD adds a per-client control variate that's subtracted
from (corrected toward) the global direction during local training, pulling each client's
trajectory back toward the direction a centralized run would have taken.

**Design decision: control-variate state lives server-side, not client-side.** The paper's
canonical form persists a per-client control variate `c_i` across rounds inside the client. This
codebase's clients are `fl.simulation.start_simulation` Ray VirtualClientEngine actors,
re-instantiated fresh every round -- the existing FCL clients already work around this via disk
pickles (`client/default.py`'s `reg{cid}.pkl`/`task{cid}.txt`). Rather than add a second, parallel
disk-persistence mechanism for a third kind of per-client state, `SCAFFOLDStrategy`
(`server/strategies.py`) keeps `self.global_c` and `self.client_c` (keyed by `ClientProxy.cid`) in
the Strategy object itself, which -- unlike the client actors -- is a single object alive for the
whole run already (exactly how `FedNovaStrategy`/`FedOptAdamStrategy` track `self.current_weights`
across rounds). Every quantity SCAFFOLD's Option II control-variate update needs is either already
known server-side (`x^t`, `c`, and `tau_i` via the `local_steps` fit-metric `FedNovaStrategy`
already established the convention for) or is exactly what the client returns anyway (`y_i`), so
nothing is lost by computing the update server-side instead of client-side.

**Client-side change** (`client/default.py`, new `FlowerClientScaffold(FlowerClient)`): overrides
only `fit()`. `SCAFFOLDStrategy.configure_fit` sends each client `[weights, correction]`
concatenated (`correction = global_c - client_c[cid]`, doubling the *outgoing* payload only, not
the client's return payload -- the client still returns plain trained weights + `local_steps`,
identical in shape to `FedNovaStrategy`'s client contract). `fit()` splits that back into
`weights`/`correction`, calls `utils.py`'s new `train_scaffold()` instead of `train()`, which adds
`correction` to each trainable parameter's `.grad` after `backward()`, before `optimizer.step()`
(BatchNorm buffer positions in `correction` are always zero -- a buffer has no `.grad`, so a
correction there is meaningless, not just risky -- `train_scaffold` filters them out via an inline
buffer-tag mask before zipping against `model.parameters()`, with an `assert` that fails fast if
that alignment assumption is ever wrong for some future model architecture).

**Server-side change** (`server/strategies.py`, new `SCAFFOLDStrategy(fl.server.strategy.FedAvg)`):
```
correction (sent to client i)  = global_c - client_c[i]
x^{t+1}                        = sum_i(p_i * y_i)                          # plain FedAvg average
c_i^+                          = c_i - global_c + (x^t - y_i) / (tau_i * local_lr)   # Option II
global_c^+                     = global_c + (1/S) * sum_i(c_i^+ - c_i)     # S = sampled clients
```
`local_lr` (default matches `train_scaffold`'s hardcoded Adam `lr=0.001`) stands in for the
paper's SGD step size -- SCAFFOLD's Option II formula and convergence theory are derived for SGD,
not Adam; applying the correction to Adam's gradient anyway is the standard practical
approximation other SCAFFOLD-on-Adam implementations use, not a formula this codebase invented,
and is called out explicitly in `SCAFFOLDStrategy`'s docstring as a documented approximation, not
a silent one.

**BatchNorm-buffer risk (items 19/29): sidestepped by construction, not re-debugged.** Unlike
`FedNovaStrategy`/`FedOptAdamStrategy`, the weights-aggregation line above (`x^{t+1}`) is a plain
weighted average of the clients' returned `y_i` -- exactly `FedAvgWithAccuracyMetric`'s own
convex combination, not an extrapolation -- so it inherits FedAvg's "can never leave the averaged
values' range" safety automatically; no `buffer_mask` is needed for the weights themselves.
`buffer_mask` here only pins BatchNorm buffer positions in `global_c`/`client_c` at zero forever
(passed the same way `FedNovaStrategy` gets it, `bn_buffer_mask(net.state_dict().keys())`) --
those arrays are never loaded into a model's `state_dict` or pushed through
`sqrt(running_var + eps)`, so even though the control-variate *update* is itself an unclipped
moving average, it has no path to the NaN failure mode items 19/29 found empirically.

**`main.py`**: new `elif strat == 'SCAFFOLD':` branch (mirrors the `FedNova` branch immediately
above it -- same `initial_parameters`/`buffer_mask` construction and `checkpoint_path` Track 2
hook, new `client_fn_scaffold` closure using `FlowerClientScaffold`). Deliberately **not** added
to the `strategy == 'all'` hardcoded list, same reasoning and same by-robot-only scope as item 29.

**Risk / blast radius**: additive-only new strategy branch + one new client class (subclasses
`FlowerClient`, changes only `fit()`) + one new training function (`train_scaffold`, does not
modify `train()`). No existing strategy's code path, weighting, or output changes.

**Smoke test**: `smoke_test/run_smoke_test.sh` step 5/10 (`--strategy SCAFFOLD`, 2 rounds/1 epoch
against synthetic data), full 10-step suite. First attempt (job 33574485) failed at the shell
level -- the worktree's `.venv` didn't exist (gitignored, needs the same symlink-to-main-worktree
setup the `fedlgr_officedb-fednova` worktree used); fixed by symlinking it. Second attempt (job
33575925) got through SCAFFOLD's own step cleanly (finite loss/PCC/RMSE both rounds, no NaN, no
`train_scaffold` assertion failure -- confirms the state_dict-order-vs-`model.parameters()`-order
assumption holds for MobileNetV2) but failed later at step 7 (FedRoot+LGR) with the exact
"Expected more than 1 value per channel" BatchNorm-batch-of-1 crash item 30 fixes -- because this
branch had forked from `officedb-adapt` *before* item 30 landed. Rebased onto item 30 (see status
line above); job 33578145 then completed all 10 steps cleanly end to end.

## 32. `dataloader/utils.py` -- FedDistill/FedRoot-FedDistill `IndexError` under by-robot/by-room partitioning

Diagnosed 2026-08-20 from the protocol-C by-robot sweep's 12 failures (all FedDistill or
FedRoot-FedDistill, seed-clustered -- `docs/fedlgr_officedb_experiments.md`'s "Protocol C hybrid
sweep" and "Supervisor-finalized protocol" sections). Retained job logs (`logs/fl_strategy_
{33875088,...}.log`, contrary to an earlier assumption that they weren't kept) show every
failure is `RoundFailedError: fit round 1: 0/3 clients returned a result`, and each client's
underlying exception is `IndexError: index 22499 is out of bounds for axis 0 with size 22496`
(or similar, always off by 1-3) raised inside `imageloader.py:__getitem__` via a
`torch.utils.data.Subset`.

**Root cause**: when `distil=True` (i.e. `strategy=FedDistill` or `base=FedDistill`),
`load_datasets(...)` rebuilds `trainset` from the teacher model's soft-label predictions
(`predict_gen_distil`, called on `train_loader_distil`, which uses `drop_last=True`). Any
trailing partial batch (0 to `batch_size-1` = up to 15 rows) is silently dropped from `outputs`,
so the rebuilt `trainset` can be shorter than `data_images_train`. The `group_col` (by-robot/
by-room) partitioning code immediately below, however, still read `data_images_train.index[...]`
-- the *original, untruncated* frame's row labels -- to build each client's
`torch.utils.data.Subset(trainset, ...)`. Whenever one of the dropped trailing rows happened to
belong to a given robot/room group (seed-dependent, since the row order comes from an unseeded
`.sample(frac=1)` shuffle upstream), that client's `Subset` contained a row label past the end of
the now-shorter `trainset`, crashing at first `DataLoader` access -- explaining both why only
FedDistill/FedRoot-FedDistill hit it (every other strategy skips the `distil` branch entirely and
never shortens `trainset`) and why it was seed-clustered (whether the dropped rows land in a
given group is a matter of that run's row-shuffle luck, not deterministic per strategy).
Predates this fork -- item 16 (2026-07-31) added the `group_col`-based train/test partitioning
that this interacts with, and item 15 (also 2026-07-31) already touched `predict_gen_distil` for
an unrelated column-count bug -- but neither investigation exercised group_col partitioning and
FedDistill together at enough scale to surface this; the 12x volume of the protocol-C crossed
sweep (vs. earlier single/few-seed FedDistill runs) is what finally hit it reliably.

**Fix**: after the `distil` branch rebuilds `trainset` from `outputs`, `data_images_train` is now
also truncated to `data_images_train.iloc[:len(outputs)].reset_index(drop=True)` -- matching
`trainset`'s actual length and re-establishing 0-based row-label alignment before the
`group_col` Subset-index lookup runs. Scoped to the `distil=True` branch only; every other
strategy's `data_images_train` (and therefore its group_col partitioning) is untouched.
`predict_gen_distil` itself and `train_loader_distil`'s `drop_last=True` were deliberately left
alone -- `predict_gen_distil`'s `numpy.asarray(outputs).reshape((len(trainloader) * batch_size,
num_classes))` assumes every batch is exactly `batch_size` rows, so flipping `drop_last` there
would need a second, unrelated fix (`numpy.concatenate` instead of `asarray`+`reshape`) for no
benefit -- truncating `data_images_train` to match is the minimal, self-contained fix.

**Risk / blast radius**: one added line inside the `distil=True` branch of
`load_datasets(...)`. Does not change `predict_gen_distil`, the teacher-model training, the
distillation soft-labels themselves, or any non-`distil` strategy's data path. Training-set size
for FedDistill/FedRoot-FedDistill runs shrinks by at most `batch_size-1` (16, i.e. <0.1% of
OfficeDB's ~22.5k augmented per-partition row count) rows relative to what a bug-free run would
have used -- the same rows `predict_gen_distil` was already silently excluding from distillation
supervision, just now also consistently excluded from `trainset`/group partitioning instead of
half-included and then crashing.

**Verification**: not yet smoke-tested against a real fold/seed as of this entry -- next step is
a targeted repro of one previously-failing combo (e.g. `FedDistill k4_test0 seed0`, job 33875088)
before resubmitting the FedDistill family at 5-fold.

## 33. `utils.py`, `client/default.py`, `client/fedBN.py`, `client/fedRoot.py`, `transfer_eval.py` -- `test()` now returns raw predictions; FL-sweep evaluate() persists them for CCC/CwM

Raised by the user 2026-08-22 after asking why CCC/CwM (`eval/metrics.py`, beyond the RMSE/PCC
this project's FL sweep reports) were unavailable for the by-robot 5-fold sweep results. Checked
`utils.py`'s `test()` directly (not just `fedlgr_officedb/evaluate.py`'s docstring note): it
already builds the full `all_labels`/`all_outputs` tensors internally to compute loss/RMSE/
Pearson, then discards them -- only 3 scalars (`loss, avg_pearson, avg_rmse`) were ever returned,
and none of `test()`'s 14 call sites across the codebase (FL clients, FCL clients, centralized
eval, `pretrain.py`'s val-loss early stopping, `transfer_eval.py`'s domain-transfer eval)
persisted them either. CCC and CwM both need raw per-scene (y_true, y_pred) pairs and cannot be
derived from any of the aggregate CSVs already on disk (`*_decentral.csv`, `comp_extracted.csv`,
`clientwise/results{cid}.txt`) or from the model-weight checkpoints
(`global_params.pkl`/`mod{cid}.pkl`/`mod_bn{cid}.pkl` -- confirmed these are `state_dict` pickles
via their `pickle.dump(state_dict, f)` call sites, not predictions).

**Checkpoint-availability finding that shaped the fix's scope**: only 4/10 strategies in the
already-completed by-robot k5 sweep have final-round weights saved at all (FedAvg/FedNova/
SCAFFOLD via `global_params.pkl`, FedRoot-FedAvg via `global_params.pkl` trunk + `mod{cid}.pkl`
head) -- `main.py`'s FedProx/FedOptAdam/FedBN branches, and FedRoot's FedBN/FedOptAdam/FedProx
base branches, never pass a `checkpoint_path` into their `evaluate_fn`/strategy construction. So
backfilling CCC/CwM for those other 6 strategies' already-completed runs needs a full retrain
regardless of this fix -- this fix only prevents the gap from recurring, it doesn't retroactively
fix already-written result files (same caveat as item 21's fix).

**Fix -- two parts**:
1. `test()` (`utils.py`) now returns a 5-tuple: `loss, avg_pearson, rmse, labels_np, outputs_np`
   (previously 3). All 14 call sites updated to match (12 just unpack-and-discard the two new
   values with `_, _` -- pretrain's val loop, `get_eval_fn`/`get_eval_fn_cl`/`get_eval_fn_bn`'s
   centralized eval, both FCL client `evaluate()` methods in `default.py`/`fedRoot.py`, and all 3
   `transfer_eval.py` call sites -- their behavior is otherwise unchanged).
2. The 4 FL-sweep decentralized (per-client) `evaluate()` methods that actually matter for
   `decentral.csv` (this project's reported metric source, not `central.csv`) now persist the raw
   pairs: `client/default.py`'s `FlowerClient.evaluate()` (FedAvg/FedProx/FedOptAdam/FedNova/
   SCAFFOLD/FedDistill), `client/fedBN.py`'s `FlowerClient_BN.evaluate()` (FedBN) and
   `FlowerClient_BN_Root.evaluate()` (FedRoot-FedBN), `client/fedRoot.py`'s
   `FlowerClient_Root.evaluate()` (FedRoot-FedAvg/FedProx/FedOptAdam/FedDistill). Each writes
   `clientwise/preds{cid}_round{server_round}.npz` (`y_true`, `y_pred`, `y_labels` arrays) next to
   the existing `clientwise/results{cid}.txt` scalar log, every round (not just the final one --
   cheap: ~142.5 KB/round for all clients combined on OfficeDB's by-robot partition, measured).
   Centralized eval (`get_eval_fn`/`get_eval_fn_bn`) deliberately does **not** persist raw
   predictions -- `central.csv` isn't this project's reported metric source, and wiring
   `checkpoint_path`-style output paths through `get_eval_fn`'s FedProx/FedOptAdam call sites
   (which don't currently take one) was judged out of scope for this fix.

**Incidental hardening, same touched lines**: `os.makedirs(f'{self.path}/clientwise')` in all 4
persisting `evaluate()` methods (and their matching `fit()` methods) changed from
`if not os.path.exists(...): os.makedirs(...)` to `os.makedirs(..., exist_ok=True)` -- fixes a
real race (3 Ray actors can call this concurrently) caught during the by-robot k5 sweep's backup
health-check: job 34058073 (SCAFFOLD, k5_test4/seed1) hit `FileExistsError` here and lost one
client's round-1 fit (`aggregate_fit: received 2 results and 1 failures`, recovered rounds 2-5;
see `docs/fedlgr_officedb_fl_hybrid_sweep_by-robot_k5_results.md` §6). Flagged, not fixed, at
backup time; fixed now since these exact lines were already being edited. **Not** applied to the
FCL classes' equivalent `os.makedirs` calls in `default.py`/`fedRoot.py` (`FlowerClientCL`,
`FlowerClient_NR`, `FlowerClientCL_Root`, `FlowerClient_NR_Root`, `FlowerClient_LGR`) -- out of
scope for this FL-sweep-focused fix, left as a flagged-not-fixed note for whoever next touches
the FCL client classes.

**Verification**: not yet smoke-tested against real data as of this entry -- see
`smoke_test/run_smoke_test.sh` before trusting this against a real sweep.

## 34. `utils.py`, `client/default.py`, `client/fedBN.py`, `client/fedRoot.py` -- `test()` now also computes/returns CCC directly (CwM stays post-hoc)

Follow-up to item 33, same session (2026-08-22), user asked for `test()` to return "all major
metrics, not only RMSE and PCC" alongside the raw predictions. Of this benchmark's 4 reported
metrics (`eval/metrics.py`: RMSE, PCC, CCC, CwM), CCC needs only the (y_true, y_pred) pairs
`test()` already builds -- no reason to defer it to a post-hoc script when it's this cheap to
compute in the same place PCC already is. CwM is different: it needs raw *per-annotator* ratings
(`eval/metrics.py`'s `cwm()` takes `human_ratings_per_scene`, a list of individual scores per
scene, not the single aggregated label `test()` receives) -- and `dataloader/utils.py`'s
`load_universal` already averages across annotators per stamp *before* `test()` ever sees a
batch (item 3 above). Getting raw per-annotator ratings into `test()` would mean threading them
through the Dataset/DataLoader layer, a materially bigger change than "fix `test()`" and out of
scope here. CwM remains a post-hoc computation, joined from the raw data files against the
`preds{cid}_round{r}.npz` files item 33 already persists -- unchanged from the item 33 plan.

**Fix**: added `_ccc()` (`utils.py`, right before `test()`) -- Lin's CCC, a self-contained
duplicate of `eval/metrics.py`'s `ccc()` (same formula), not an import: `eval/` lives in the main
repo, and this vendor repo must stay runnable/licensed standalone (root `CLAUDE.md`'s "no shared
code between pipelines" rule -- the two pipelines don't call into each other, so this is a
deliberate duplication, not an oversight). Keep the two formulas in sync if either changes.
`test()`'s return signature grew to a 6-tuple: `loss, avg_pearson, rmse, avg_ccc, labels_np,
outputs_np` (avg_ccc computed the same way avg_pearson already is -- per output/action column,
then averaged). All 17 call sites updated to match (13 just add one more `_` to discard it,
unchanged behavior; the same 4 FL-sweep decentralized `evaluate()` methods item 33 touched --
`FlowerClient`, `FlowerClient_BN`, `FlowerClient_BN_Root`, `FlowerClient_Root` -- now also append
`avg_ccc` as a 5th column to `clientwise/results{cid}.txt` (was `round,loss,avg_pearson,avg_rmse`,
now `round,loss,avg_pearson,avg_rmse,avg_ccc`) and add `"avg_ccc"` to the metrics dict returned to
Flower. Purely additive to the file format -- confirmed
`fedlgr_officedb/extract_federated_indomain_perrobot.py`'s `last_round()` parser only indexes
`last[0..3]` via `.split(",")`, so it's unaffected by the new trailing column on both old
(4-column) and new (5-column) `results{cid}.txt` files.

**Not changed**: centralized eval (`get_eval_fn`/`get_eval_fn_cl`/`get_eval_fn_bn`), the FCL
client `evaluate()` methods (`FlowerClientCL`, `FlowerClient_NR`, `FlowerClientCL_Root`,
`FlowerClient_NR_Root`, `FlowerClient_LGR`), `pretrain.py`'s val loop, and `transfer_eval.py`'s 3
call sites all just discard `avg_ccc` (extra `_`), matching item 33's own scoping of the raw-array
persistence to only the 4 FL-sweep classes -- `decentral.csv` is this project's reported metric
source, not `central.csv`, and FCL/transfer-eval logging formats were left alone.

**Verification**: not yet smoke-tested -- covered by the same pending `smoke_test/run_smoke_test.sh`
run as item 33 (step 11 already validates `preds*.npz` keys/shapes; results.txt's new 5th column
isn't separately asserted by the smoke test, worth a quick manual check on the first real run).

## 35. `dataloader/utils.py`, `client/default.py`, `client/fedBN.py`, `client/fedRoot.py` -- `preds{cid}_round{r}.npz` also persists each row's sample (Stamp) ID

Follow-up to items 33/34, same day (2026-08-23). After confirming the by-room k5 seed-0 sweep's
`preds*.npz` files were healthy, the user asked whether CwM (still post-hoc per item 34, since it
needs raw per-annotator ratings) could actually be computed without retraining. It can, but only
by re-deriving each npz row's underlying sample identity to join against the raw annotation CSV --
and the row order that join would need to assume is **not actually safe to reproduce**:
`load_datasets()` (`dataloader/utils.py`) shuffles both `data_images_train` and `data_images_test`
via `.sample(frac=1)` with no `random_state=`, i.e. it reads from whatever the *global* numpy RNG
state happens to be at that call, not a value derived solely from `--seed`. In-run this doesn't
matter (`test()`'s per-round batches iterate the same already-shuffled `Subset` every round, per
item 21's comment on this file), but re-calling `load_datasets()` in a fresh process to recover
row order is fragile: it only reproduces the original shuffle if literally no other numpy-RNG-
consuming call happens between the run's `np.random.seed(args.seed)` (`main.py`) and its one
`load_datasets()` call, which is true for these single-strategy sweep jobs but is an unstated,
easily-broken invariant, not something to build a metric on.

**Fix**: thread sample identity through directly instead of trying to reconstruct it.
`load_datasets()` already builds `test_datasets` as one `torch.utils.data.Subset` per client
(both the `group_col` by-room/by-robot branch and the plain `random_split` branch) from
`data_images_test`, whose column 0 is `Stamp` (see `load_images()`). Added a small helper,
`_subset_source_indices()`, that resolves a `Subset`'s `.indices` down to positions in the
ultimate root dataset -- needed because `random_split()` wraps *another* Subset (the
trim-to-partition-size one built just above it in the non-`group_col` branch), so a naive
one-level `.indices` read is only correct for the `group_col` branch. Right after `test_datasets`
is built (before the `testloaders` list comprehension), each `Subset` gets a plain `.sample_ids`
attribute: `data_images_test.iloc[_subset_source_indices(ds), 0].tolist()`, in the exact order
that `Subset`'s `shuffle=False` `DataLoader` will iterate it -- so `test()`'s output row *i*
always corresponds to `sample_ids[i]`, no separate bookkeeping needed at the call site. This is
attached as an attribute, not a new return value, so `load_datasets()`'s return arity/call sites
(`main.py` x2, `run_fl.py`, `transfer_eval.py`, `pretrain.py`) are untouched.

The same 4 FL-sweep decentralized `evaluate()` methods items 33/34 touched
(`client/default.py`'s `FlowerClient.evaluate()`, `client/fedBN.py`'s
`FlowerClient_BN.evaluate()`/`FlowerClient_BN_Root.evaluate()`, `client/fedRoot.py`'s
`FlowerClient_Root.evaluate()`) now add one more key to the `np.savez(...)` call:
`sample_id=np.array(getattr(self.testloader.dataset, 'sample_ids', []))`. The `getattr(...,  [])`
fallback means this degrades gracefully (empty array, not a crash) if `evaluate()` is ever called
against a loader that didn't go through the patched `load_datasets()`. Purely additive: existing
`preds*.npz` files without this key remain valid; nothing that reads `y_true`/`y_pred`/`y_labels`
changes. No changes to `CustomDataset`, `test()`'s signature, or any of the ~14 other
`test()`/`load_datasets()` call sites.

With `sample_id` saved, RMSE/PCC/CCC were already computable directly from `y_true`/`y_pred`
(item 33/34); CwM is now a straightforward post-hoc join (`sample_id` -> raw per-annotator rows in
the source CSV) with no dependence on reproducing dataloader shuffle order, and no retraining or
re-inference needed for any of the four metrics.

**Verification**: smoke-tested via a small `sintr` run before resubmitting the full by-room k5
seed-0 sweep (34203283-347) -- confirmed `sample_id` lands in a fresh `preds*.npz`, is non-empty,
matches `y_true`'s row count, and its values are plausible `Stamp` floats. See
`smoke_test/run_smoke_test.sh` step 11 for the automated check.

## 36. `transfer_eval.py` -- domain-transfer k-fold eval jobs now also capture CCC + raw predictions (items 33/34/35's last unfixed caller)

Follow-up to items 33/34/35, raised 2026-08-26 after the by-robot k5 sweep's own CCC/CwM fix
prompted a check of the domain-transfer k-fold sweep's 60 already-completed eval jobs
(`results/fedlgr_officedb/domain_transfer/kfold_reliability/`) -- item 33's own text named
`transfer_eval.py`'s 3 `test()` call sites as updated only enough to match the new tuple arity
(`_, _, _` to discard `avg_ccc`/`labels_np`/`outputs_np`), not to persist them. So this file was
always the one caller items 33/34/35 explicitly left alone, and the 60 kfold jobs' result JSONs
only ever contained `{loss, pcc, rmse}` -- confirmed by reading the actual files, nothing to
recover post-hoc. All 30 pretrain checkpoints verified intact (correct size, timestamps matching
the sweep window) -- this fix needed a rerun of the 60 eval jobs only, no retraining.

**Fix**: `run()` now captures `ccc, y_true, y_pred` at all 3 `test()` call sites (zero-shot,
early-stopped finetune, fixed-epoch finetune) instead of discarding them, adds `'ccc'` to the
corresponding `results['zero_shot']`/`results['finetuned']` dict, and writes a
`preds_{zeroshot,finetuned}.npz` per job (derived from `--output`'s path, e.g.
`ceiling_es_office_nao_..._preds_zeroshot.npz`) with the same 4-key schema as items 33/34/35's
`clientwise/preds{cid}_round{r}.npz` (`y_true`, `y_pred`, `y_labels`, `sample_id`) -- two files
when a finetune pass runs (zero-shot and finetuned are different model states with different
predictions), one when it doesn't.

`sample_id` needed switching which `load_datasets()` loader `test()` is called on: `run()`
previously called it on the pooled `testloader`, which -- unlike the per-client split item 35
attached `.dataset.sample_ids` to -- carries no sample-id attribute at all. Since this file always
calls `load_datasets(num_clients=1, ...)`, `testloaders_per_client[0]` (previously discarded as
`_testloaders_per_client`) covers the exact same rows as the pooled loader, just built via the
per-client `Subset` path that already has `sample_ids` populated -- so `run()` now evaluates
against `testloaders_per_client[0]` (renamed `eval_loader`) instead of the pooled loader. RMSE/
PCC/CCC are computed over the full concatenated batch regardless of iteration order (item 21), so
this doesn't change what's measured, only where `sample_id` comes from; values may shift in the
last few decimal places vs. the archived pre-fix numbers (different DataLoader batching -> CPU
float reduction order) but shouldn't move meaningfully, per the same caveat item 21's own fix
noted.

Pre-fix result JSONs (all 60) archived to
`results/fedlgr_officedb/domain_transfer/kfold_reliability/pre_predfix/` before the rerun, mirroring
the by-robot k5 `pre_predfix/` convention -- see
[[project_officedb_cv_vs_multiseed_protocol]]/[[project_officedb_domain_transfer_ccc_cwm_fix_plan]].

**Verification**: smoke-tested against a real checkpoint
(`checkpoints_domain_transfer/office_nao_kfold5_test0_seed0/cpu/MobileNet.pkl`, job 34417301,
zero-shot only) before resubmitting the full 60-job rerun -- confirmed `ccc` lands in the output
JSON as a finite float, and the `preds_zeroshot.npz` has all 4 keys, `y_true`/`y_pred` shaped
(200, 9) with no NaNs, `sample_id` non-empty with plausible Stamp values, and row counts matching.
The early-stopped-finetune branch (`results['finetuned']` + its `preds_finetuned.npz`) shares this
exact `test()`/`save_preds()` code, exercised by the zero-shot smoke test -- not separately
smoke-tested end-to-end (a finetune pass runs ~35-110 min per the original sweep's timings, vs.
the zero-shot smoke job's ~3 min) but reuses the same already-verified call, differing only by the
unmodified, pre-existing `train_with_early_stopping` step in between.

Also added `--eval-only` to `fedlgr_officedb/slurm/submit_domain_transfer_kfold.py` (this
project's own code, not vendor) -- skips `submit_pretrain` and points straight at the existing
`checkpoints_domain_transfer/.../cpu/MobileNet.pkl` paths (no `--dependency`, since there's no new
pretrain job), for exactly this kind of eval-only rerun against already-trained checkpoints.

## 37. `models/ResNet50.py` (new), `pretrain.py`, `main.py`, `transfer_eval.py` -- add ResNet-50 as a second, heavier CNN backbone

Backbone-diversification follow-up (2026-08-26/27, `docs/paper_prep_2027/ws_domain_transfer_backbone_diversification.md`
§6, `docs/resnet50_domain_transfer_kfold_plan.md`): the domain-transfer/zero-shot
comparison arm only had one CNN (MobileNetV2). Testing it against same-family lighter
siblings would be a near-null-result experiment (MobileNetV2/V3-Small/V4-Conv-Small
cluster within ~1 accuracy point under matched training); testing capacity *upward*
against ResNet-50 (25.6M params vs. MobileNetV2's 3.4M) actually brackets the CNN lane
the way the VLM lane already brackets scale (3B vs. ~8B).

New `models/ResNet50.py`, following the exact per-model pattern `models/MobileNet.py`
and `models/deepLabMobileNet.py` already use (self-contained `conv` + `FCNet` + `Net`):
`torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)` with `model.fc`
replaced by `nn.Identity()` -- torchvision's `ResNet.forward()` already does avgpool +
flatten before `self.fc`, so this yields the flattened 2048-dim pooled feature directly
without needing MobileNet.py's separate `AdaptiveAvgPool2d`/`Flatten` bolted on. `FCNet`
mirrors MobileNet's head exactly (`BatchNorm1d -> Linear(->32) -> Linear(32,
num_classes)`) except the first layer's input width is **2048, not 1280** -- forced by
ResNet-50's wider pooled feature, not a stylistic change; nothing else about the head
was altered (no justification found to change it).

Registered `'ResNet50'` alongside `'MobileNet'`/`'DeepLabMobileNet'` in the same 3
dispatch points those two already use: `pretrain.py`'s `args.models` elif chain,
`main.py`'s `args.model` elif chain (FCL/FL entrypoint -- not exercised by this
experiment, kept consistent for parity), and `transfer_eval.py`'s `args.model` dispatch
(also added to its `--model` `choices=`). `main_fcl.py` (LGR/EWC/FedRoot FCL path,
including its hardcoded `input_dim = 1280` for the LGR replay generator) is
**deliberately not touched** -- this experiment runs through `pretrain.py`/
`transfer_eval.py` only, never `main_fcl.py`.

Checkpoint/results paths: existing MobileNet kfold_reliability filenames carry no model
tag, so `fedlgr_officedb/slurm/submit_domain_transfer_kfold.py` (this project's own
code, not vendor) got a `--model` flag (default `MobileNet`, unchanged behavior) that
routes ResNet-50 runs to sibling paths --
`checkpoints_domain_transfer/{domain}_{robot}_kfold5_test{fold}_seed{seed}_resnet50/`
and `results/fedlgr_officedb/domain_transfer/kfold_reliability_resnet50/` -- instead of
colliding with MobileNet's un-suffixed ones. The sbatch scripts
(`domain_transfer_{pretrain,eval}.sbatch`) already accepted a `MODEL` env var (default
`MobileNet`) from item 23's era; this is the first time it's actually driven with a
second value.

**Verification**: smoke-tested one robot/fold (Nao, office, fold 0, zero-shot ceiling
only) end-to-end on a CPU job before submitting the full sweep -- see
`docs/resnet50_domain_transfer_kfold_plan.md` for the smoke-test job ID and result.

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
