import pandas as pd
import os
import torchvision.transforms as transforms
from .imageloader import CustomDataset
from torch.utils.data import random_split, DataLoader
import torch
import numpy as np
import sys
from sklearn.utils import shuffle

sys.path.append('..')
from utils import predict_gen, train, predict_gen_distil


# Default column layout matches vendor's original MANNERS-DB `all_data.csv`
# (see OFFICEDB_MODIFICATIONS.md). `load_universal`/`load_images` accept
# overrides so OfficeDB's 9-action, non-circle/arrow layout can pass straight
# through instead of being shimmed into these literal names.
_DEFAULT_ACTION_COLS = [
	'Vacuum cleaning', 'Mopping the floor', 'Carry warm food', 'Carry cold food',
	'Carry drinks', 'Carry small objects (plates, toys)', 'Carry big objects (tables, chairs)',
	'Cleaning (Picking up stuff) / Starting conversation',
]
_DEFAULT_EXTRA_COLS = ['Using circle', 'Using arrow']

# Every DataLoader below that wraps CustomDataset does real per-sample disk
# I/O (Image.open in imageloader.py's __getitem__), and none of them
# previously set num_workers (default 0 = fully synchronous, single-process
# loading, no prefetching). Confirmed via a direct timing check (see
# fedlgr_officedb_gpu branch/project memory) that real image reads on this
# cluster's shared Lustre currently take ~85-115ms each -- with num_workers=0
# that's fully serialized against training, dwarfing the actual (sub-second)
# GPU/CPU compute per batch. 2 workers is deliberately conservative, not
# tuned: chosen to fit within the smallest per-client CPU budget in use
# (4 CPUs/client on GPU jobs, see submit_fl_sweep.py's COMPUTE dict) while
# leaving headroom for the main process. Purely a perf change -- does not
# affect training results (same data, same order via shuffle=..., just
# loaded by worker subprocesses instead of the main process).
NUM_WORKERS = 2


def _subset_source_indices(subset):
	"""Resolve a (possibly nested) torch Subset's .indices down to positions
	in its ultimate root dataset. random_split() wraps ANOTHER Subset (the
	trimmed-to-partition-size one built just above it), so a naive one-level
	`.indices` read is only correct for the group_col branch's single-level
	Subsets -- this also handles the random_split branch's nesting, so
	sample-id lookup (OFFICEDB_MODIFICATIONS.md item 35) is correct either
	way without relying on random_split's inner Subset happening to be an
	identity range."""
	indices = list(subset.indices)
	base = subset.dataset
	while isinstance(base, torch.utils.data.Subset):
		indices = [base.indices[i] for i in indices]
		base = base.dataset
	return indices


def load_universal(path, stamp_col='Stamp', mean_cols=None):
	data = pd.read_csv(path + "/all_data.csv")
	# mean_cols are averaged across annotators per stamp (original behaviour);
	# everything else (group/task/split id columns, which are constant within
	# a stamp) is carried through via 'first' instead of 'mean', since those
	# can be non-numeric (e.g. split='train') and .mean() can't handle that.
	# Original code meant every column via a range(250, 1000) window, which
	# both hardcoded a 3-digit-stamp assumption and silently capped capacity
	# at 750 stamps -- too few for OfficeDB's 1000 images/robot. Iterating the
	# actual unique stamps removes both limitations.
	if mean_cols is None:
		mean_cols = [c for c in data.columns if c != stamp_col]
	agg = {c: ('mean' if c in mean_cols else 'first') for c in data.columns if c != stamp_col}
	df = data.groupby(stamp_col, as_index=False).agg(agg)
	return df


def load_augmented(df):
	data_new = pd.DataFrame(columns=df.columns)
	for c in range(10):
		for i in range(df.shape[0]):
			data_new = pd.concat([data_new, df.iloc[i].to_frame().T])
	return data_new


def sort_nicely(l):
	""" Sort the given list in the way that humans expect.
	"""
	import re
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	l.sort(key=alphanum_key)
	return l


def load_images(path, action_cols=None, extra_cols=None, stamp_col='Stamp'):
	# action_cols/extra_cols default to MANNERS-DB's original 8 actions +
	# circle/arrow columns; OfficeDB adaptation passes its own 9 action names
	# and group/task/split id columns instead (see OFFICEDB_MODIFICATIONS.md).
	action_cols = list(action_cols) if action_cols is not None else list(_DEFAULT_ACTION_COLS)
	extra_cols = list(extra_cols) if extra_cols is not None else list(_DEFAULT_EXTRA_COLS)

	df = load_universal(path, stamp_col=stamp_col, mean_cols=action_cols)
	imgpath = path + "/images"
	li = sort_nicely(os.listdir(imgpath))
	columns = [stamp_col, 'path'] + extra_cols + action_cols

	rows = []
	for fname in li:
		try:
			# Original sliced the first 3 filename characters (`fname[:3]`),
			# which silently mis-parses any stamp >= 1000 (e.g. "1000_..."
			# -> "100"). Splitting on the first underscore reads the whole
			# leading integer regardless of digit width -- needed since
			# OfficeDB pools can exceed 999 distinct stamps (e.g. by-robot's
			# 3000-image combined pool). Exact match for the <1000 case.
			stamp = float(fname.split('_')[0])
			row = df[df[stamp_col] == stamp]
			entry = [stamp, f'{imgpath}/' + fname]
			entry += [row[c].iloc[0] for c in extra_cols]
			entry += [float(row[c].iloc[0]) for c in action_cols]
			rows.append(entry)
		except Exception:
			print('Error Loading FileName; Continuing to next.')

	return pd.DataFrame(rows, columns=columns)


def load_datasets(num_clients, path, aug, batch_size=16, out='', DEVICE=torch.device("cpu"), data_permutation=None,
				  distil=False, teacher_model=None, action_cols=None, extra_cols=None, split_col=None, group_col=None):
	# action_cols/extra_cols: see load_images. split_col: if given and present
	# in the built dataframe, honor its train/test labels instead of the
	# internal random 75/25 split (lets this reuse data/splits/*.csv's
	# leakage-safe 80/10/10 splits; 'val' rows are simply not consumed here,
	# since this training loop -- like vendor's original -- has no validation
	# phase). group_col: if given, assign each client ALL rows of one unique
	# group value (robot or room) instead of blind random_split -- needed for
	# by-robot/by-room non-IID partitioning. Both are additive: omitting them
	# reproduces the original random-permutation/random-split behaviour
	# exactly. See OFFICEDB_MODIFICATIONS.md.
	action_cols = list(action_cols) if action_cols is not None else list(_DEFAULT_ACTION_COLS)
	extra_cols = list(extra_cols) if extra_cols is not None else list(_DEFAULT_EXTRA_COLS)
	label_start = 2 + len(extra_cols)

	data = load_images(path, action_cols=action_cols, extra_cols=extra_cols)

	if split_col is not None and split_col in data.columns:
		data_images_train = data[data[split_col] == 'train'].reset_index(drop=True)
		data_images_test = data[data[split_col] == 'test'].reset_index(drop=True)
	else:
		if data_permutation is None:
			data_permutation = np.random.permutation(len(data))
		data = data.iloc[data_permutation]
		# Splitting 75% Train and 25% Test Data
		data_images_train = data.iloc[:int(data.shape[0] * 0.75)].reset_index(drop=True)
		data_images_test = data.iloc[int(data.shape[0] * 0.75):].reset_index(drop=True)

	data_images_train = data_images_train.sample(frac=1).reset_index(drop=True)
	data_images_test = data_images_test.sample(frac=1).reset_index(drop=True)

	if aug:
		train_transform = transforms.Compose([
			transforms.Resize((128, 128)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(10),
			# transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
			# transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
		data_images_train = load_augmented(data_images_train)
		data_images_train = data_images_train.sample(frac=1).reset_index(drop=True)

	else:
		train_transform = transforms.Compose([
			transforms.Resize((128, 128)),  # Adjust the size according to your model requirements
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	test_transform = transforms.Compose([
		transforms.Resize((128, 128)),  # Adjust the size according to your model requirements
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	trainset = CustomDataset(dataframe=data_images_train, transform=train_transform, label_start=label_start)
	testset = CustomDataset(dataframe=data_images_test, transform=test_transform, label_start=label_start)

	if distil:
		if os.path.exists(f'{out}/teacher_model.pt'):
			teacher_model.to(DEVICE)
			teacher_model.load_state_dict(torch.load(f'{out}/teacher_model.pt'))
		elif os.path.exists(f'{out}/teacher_model_cuda.pt'):
			teacher_model.to(DEVICE)
			teacher_model.load_state_dict(torch.load(f'{out}/teacher_model_cuda.pt'))
		else:
			teacher_model.to(DEVICE)
			# trainset_distil = CustomDataset(dataframe=data_images_train, transform=train_transform)
			train_loader_distil = DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=NUM_WORKERS)
			print("Training Teacher Model")
			train(model=teacher_model, train_loader=train_loader_distil, epochs=10, DEVICE=DEVICE)
		if DEVICE.type == 'cuda':
			torch.save(teacher_model.state_dict(), f'{out}/teacher_model_cuda.pt')
		else:
			torch.save(teacher_model.state_dict(), f'{out}/teacher_model.pt')

		outputs = pd.DataFrame(predict_gen_distil(teacher_model, train_loader_distil, DEVICE))
		trainset = CustomDataset(dataframe=pd.concat([data_images_train.iloc[:len(outputs), :label_start], outputs], axis=1, ignore_index=True),
		                        transform=train_transform, label_start=label_start)
		# train_loader_distil used drop_last=True, so outputs (and the trainset
		# just rebuilt above) can be up to batch_size-1 rows shorter than
		# data_images_train. Truncate data_images_train to match -- otherwise
		# the group_col Subset-index lookup below uses row labels from the
		# untruncated frame against the now-shorter trainset and can index
		# past its end (OFFICEDB_MODIFICATIONS.md item 32).
		data_images_train = data_images_train.iloc[:len(outputs)].reset_index(drop=True)


	if group_col is not None and group_col in data_images_train.columns:
		# By-robot/by-room partitioning: each client gets every row of one
		# unique group value, instead of a blind random_split. Requires
		# exactly one client per unique group value.
		groups = sorted(data_images_train[group_col].unique())
		if len(groups) != num_clients:
			raise ValueError(
				f"group_col={group_col!r} has {len(groups)} unique values {groups} "
				f"but num_clients={num_clients} -- these must match for by-group partitioning.")
		datasets = [
			torch.utils.data.Subset(trainset, data_images_train.index[data_images_train[group_col] == g].tolist())
			for g in groups
		]
	else:
		# Split training set into `num_clients` partitions to simulate different local datasets
		partition_size = len(trainset) // num_clients

		lengths = [partition_size] * num_clients
		# trim trainset to partition_size*num_clients without iloc
		# trainset = torch.utils.data.Subset(trainset, range(partition_size * num_clients))
		datasets = random_split(torch.utils.data.Subset(trainset, range(partition_size * num_clients)), lengths, torch.Generator().manual_seed(42))
		# datasets = random_split(testset[:sum(lengths)], lengths, torch.Generator().manual_seed(42))
	y_labels = action_cols
	# Split each partition into train/val and create DataLoader
	trainloaders = []
	# valloaders = []

	for ds in datasets:
		# if distil:
		# 	outputs = pd.DataFrame(predict_gen_distil(teacher_model, DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True), DEVICE))
		# 	inputs = ds.__get_dataframe__().iloc[:, :4]
		# 	ds_teacher = pd.concat([inputs, outputs], axis=1)
		# 	trainloaders.append(DataLoader(ds_teacher, batch_size=batch_size, shuffle=True, drop_last=True))
		# 	# trainloaders.append(
		# 	# 	predict_gen_distil(teacher_model, DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True), DEVICE))
		# else:
		trainloaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=NUM_WORKERS))
	if distil:
		teacher_model = None
	# valloaders.append(DataLoader(ds_val, batch_size=batch_size))

	# Per-client test partitions, mirroring the trainset partitioning above
	# (group_col-aware if given, else a matching random_split) -- federated
	# ("distributed") evaluation is supposed to measure each client's own
	# held-out data, not the same pooled set re-evaluated N times. See
	# OFFICEDB_MODIFICATIONS.md item 16. `testloader` (pooled, unchanged) is
	# kept for the separate centralized evaluate_fn.
	if group_col is not None and group_col in data_images_test.columns:
		test_groups = sorted(data_images_test[group_col].unique())
		if len(test_groups) != num_clients:
			raise ValueError(
				f"group_col={group_col!r} has {len(test_groups)} unique test-split values {test_groups} "
				f"but num_clients={num_clients} -- these must match for by-group partitioning.")
		test_datasets = [
			torch.utils.data.Subset(testset, data_images_test.index[data_images_test[group_col] == g].tolist())
			for g in test_groups
		]
	else:
		test_partition_size = len(testset) // num_clients
		test_lengths = [test_partition_size] * num_clients
		test_datasets = random_split(
			torch.utils.data.Subset(testset, range(test_partition_size * num_clients)),
			test_lengths, torch.Generator().manual_seed(42))
	# sample_ids (OFFICEDB_MODIFICATIONS.md item 35): each per-client test
	# Subset's Stamp values, in the exact same order the (shuffle=False,
	# below) testloader will iterate them -- so test()'s row-i output lines
	# up with sample_ids[i] with no extra bookkeeping at the call site.
	# Attached as a plain attribute (not a new return value) so this doesn't
	# touch load_datasets()'s return arity / any of its other call sites.
	for ds in test_datasets:
		ds.sample_ids = data_images_test.iloc[_subset_source_indices(ds), 0].tolist()
	testloaders = [DataLoader(ds, batch_size=batch_size, num_workers=NUM_WORKERS) for ds in test_datasets]

	testloader = DataLoader(testset, batch_size=batch_size, num_workers=NUM_WORKERS)
	# return trainloaders, valloaders, testloader, y_labels
	return trainloaders, testloaders, testloader, y_labels, data_permutation


def load_val_loader(path, batch_size=16, action_cols=None, extra_cols=None, split_col='Split'):
	# Standalone val-split loader for early-stopping-based checkpoint
	# selection (OFFICEDB_MODIFICATIONS.md item 23). Deliberately independent
	# of load_datasets()'s train/test return contract -- adding a val loader
	# to that function's return tuple would require every existing call site
	# (main.py, run_fl.py, transfer_eval.py, pretrain.py) to be updated to
	# unpack one more value, for a purely additive, opt-in feature only the
	# new early-stopping path needs. 'val' rows already exist in every
	# prepared OfficeDB/MANNERSDBPlus dataset (data/splits/*.csv's 80/10/10
	# split) but were never consumed anywhere before this -- see the comment
	# on load_datasets()'s split_col branch above. Returns None if the data
	# has no split_col column or no rows labeled 'val' (e.g. a legacy
	# MANNERS-DB path with no 3-way split), so callers can fall back to the
	# old fixed-epoch behavior instead of hard-failing.
	action_cols = list(action_cols) if action_cols is not None else list(_DEFAULT_ACTION_COLS)
	extra_cols = list(extra_cols) if extra_cols is not None else list(_DEFAULT_EXTRA_COLS)
	label_start = 2 + len(extra_cols)

	data = load_images(path, action_cols=action_cols, extra_cols=extra_cols)
	if split_col not in data.columns:
		return None
	data_images_val = data[data[split_col] == 'val'].reset_index(drop=True)
	if len(data_images_val) == 0:
		return None
	data_images_val = data_images_val.sample(frac=1).reset_index(drop=True)

	val_transform = transforms.Compose([
		transforms.Resize((128, 128)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	valset = CustomDataset(dataframe=data_images_val, transform=val_transform, label_start=label_start)
	return DataLoader(valset, batch_size=batch_size, num_workers=NUM_WORKERS)


def load_datasets_pretrain(num_clients, path, split, aug=True, batch_size=16, out='', DEVICE=torch.device("cpu"),
						   data_permutation=None, action_cols=None, extra_cols=None):
	action_cols = list(action_cols) if action_cols is not None else list(_DEFAULT_ACTION_COLS)
	extra_cols = list(extra_cols) if extra_cols is not None else list(_DEFAULT_EXTRA_COLS)
	label_start = 2 + len(extra_cols)
	data = load_images(path, action_cols=action_cols, extra_cols=extra_cols)
	if data_permutation is None:
		data_permutation = np.random.permutation(len(data))
	data = data.iloc[data_permutation]
	# Splitting 75% Train and 25% Test Data
	# shuffle data dataframe
	data = shuffle(data)
	# reset index after shuffle
	data.reset_index(inplace=True, drop=True)
	# data_images_train = data.iloc[:int(data.shape[0] * (1-split))]
	data_images_test = data.iloc[int(data.shape[0] * (1 - split)):]

	test_transform = transforms.Compose([
		transforms.Resize((128, 128)),  # Adjust the size according to your model requirements
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(10),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	testset = CustomDataset(dataframe=data_images_test, transform=test_transform, label_start=label_start)
	# Split each partition into train/val and create DataLoader
	trainloaders = []
	# valloaders = []
	testloader = DataLoader(testset, batch_size=batch_size, num_workers=NUM_WORKERS)
	# return trainloaders, valloaders, testloader, y_labels
	return testloader



def ac_split(path):
	data = load_images(path)
	arrow = data[data['Using arrow'] == 1]
	circle = data[data['Using circle'] == 1]
	return arrow, circle


def task_splitter(path, task_col, task_values, n_clients, aug, batch_size=16,
				  action_cols=None, extra_cols=None, split_col=None):
	"""N-task generalization of task_splitter_circle_arrow (see
	OFFICEDB_MODIFICATIONS.md). task_values is an ordered list of any length;
	each value selects one task's rows via data[task_col] == value.

	Returns (train_per_task_cl, cumulative_test_per_task, final_test, y_labels):
	  - train_per_task_cl[t][cid]: this client's DataLoader for task t.
	  - cumulative_test_per_task[t]: DataLoader over tasks 0..t combined --
	    generalizes the paper's Task1 -> Combined(Task1+Task2) protocol to N
	    tasks (evaluate retention vs. catastrophic forgetting after each task
	    boundary, not just the final one). For the original 2-task case this
	    reproduces vendor's [test_circle, test] exactly: entry 0 is task 0
	    alone (nothing precedes it to combine with), entry -1 is all tasks
	    combined.
	  - final_test == cumulative_test_per_task[-1] (all tasks combined),
	    kept separately for callers that just want one combined test loader.
	"""
	action_cols = list(action_cols) if action_cols is not None else list(_DEFAULT_ACTION_COLS)
	extra_cols = list(extra_cols) if extra_cols is not None else list(_DEFAULT_EXTRA_COLS)
	label_start = 2 + len(extra_cols)

	if aug:
		train_transform = transforms.Compose([
			transforms.Resize((128, 128)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(10),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	else:
		train_transform = transforms.Compose([
			transforms.Resize((128, 128)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	test_transform = transforms.Compose([
		transforms.Resize((128, 128)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	data = load_images(path, action_cols=action_cols, extra_cols=extra_cols)
	y_labels = action_cols

	train_per_task_cl = []
	cumulative_test = []
	running_test_frames = []

	for value in task_values:
		frame = data[data[task_col] == value].sample(frac=1).reset_index(drop=True)

		if split_col is not None and split_col in frame.columns:
			train_frame = frame[frame[split_col] == 'train'].reset_index(drop=True)
			test_frame = frame[frame[split_col] == 'test'].reset_index(drop=True)
		else:
			n = frame.shape[0]
			train_frame = frame.iloc[:int(n * 0.75)].reset_index(drop=True)
			test_frame = frame.iloc[int(n * 0.75):].reset_index(drop=True)

		running_test_frames.append(test_frame)
		combined_test = pd.concat(running_test_frames, axis=0).sample(frac=1).reset_index(drop=True)
		cumulative_test.append(
			DataLoader(CustomDataset(combined_test, test_transform, label_start=label_start),
					   batch_size=batch_size, drop_last=True, num_workers=NUM_WORKERS))

		if aug:
			train_frame = load_augmented(train_frame)
			train_frame = train_frame.sample(frac=1).reset_index(drop=True)

		size = train_frame.shape[0] // n_clients
		client_loaders = []
		for i in range(n_clients):
			chunk = train_frame.iloc[i * size:(i + 1) * size]
			client_loaders.append(
				DataLoader(CustomDataset(chunk, train_transform, label_start=label_start), batch_size=batch_size,
						   shuffle=True, drop_last=True, num_workers=NUM_WORKERS))
		train_per_task_cl.append(client_loaders)

	final_test = cumulative_test[-1]
	return train_per_task_cl, cumulative_test, final_test, y_labels


def task_splitter_circle_arrow(path, n_clients, aug, batch_size=16):
	"""Backward-compatible wrapper reproducing vendor's original 2-task
	circle/arrow split via the generalized task_splitter above (see
	OFFICEDB_MODIFICATIONS.md). 'Using circle' and 'Using arrow' are
	complementary flags on every row (circle=1,arrow=0 or circle=0,arrow=1),
	so task_values=[1, 0] on 'Using circle' reproduces [circle-task,
	arrow-task] exactly."""
	return task_splitter(path, task_col='Using circle', task_values=[1, 0], n_clients=n_clients, aug=aug, batch_size=batch_size)


def load_datasets_hyper(path, aug):
	data = load_images(path, aug)
	data_images_train = data.iloc[:int(data.shape[0] * 0.75)]
	data_images_test = data.iloc[int(data.shape[0] * 0.75):]
	if aug == True:
		transform = transforms.Compose([
			transforms.Resize((128, 128)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(10),
			# transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
			# transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
			transforms.ToTensor(),
		])
	else:
		transform = transforms.Compose([
			transforms.Resize((128, 128)),  # Adjust the size according to your model requirements
			transforms.ToTensor(),
			#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize if required
		])
	trainset = CustomDataset(dataframe=data_images_train, transform=transform)
	testset = CustomDataset(dataframe=data_images_test, transform=transform)
	return trainset, testset
