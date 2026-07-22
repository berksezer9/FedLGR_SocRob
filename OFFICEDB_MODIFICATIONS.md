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

## 5. `main_fcl.py`

- Added `--num_classes` (default 8), `--n_tasks` (default 2), `--task_col` (default
  `'Using circle'`), `--task_values` (default `None` -> `[1, 0]`), `--pretrained_dir`
  (default `'models'`), `--action_cols`, `--extra_cols`, `--split_col`.
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

## 9. `pretrain.py`

Added `--num_classes` (default 8) and `--action_cols`, threaded to the model constructors and
`load_datasets_pretrain`. `y_labels` list generalized from the hardcoded `[0..7]` to
`list(range(args.num_classes))`.

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
