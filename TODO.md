# TODO

[ ] Add `valid_eval_args` and `test_eval_args`;

[ ] Change `eval_neg_sample_args` to `valid_neg_sample_args` and `test_neg_sample_args`;

1). used_item refers to all items that users have interacted in the training and evaluation set. positve_item refers to
all items that users have interacted in the evaluation set. history_item refers to all items that users have interacted
in the training set.

2). positive_u and positive_i are used to calculate evaluation scores. positive_u(Torch.Tensor) should be the row index
of positive items for each user in the evaluation set, and positive_i(Torch.Tensor) should be the positive item id for
each user.

# NOTE

`dataset` in config:

1. `dataset_filename` (str): The filename of dataset.
2. `dataset` (str): The name of dataset.
3. They are shown in the following form:
    - dataset/`dataset`/`dataset_filename`[.benchmark].[inter|user|item]
    - recbole/properties/dataset/`dataset`.yaml
4. NB:
    1. `dataset_filename` has a higher priority than `dataset`.
    2. If `dataset_filename` is None, `dataset` will be used as dataset name.

# Installation

```bash
git pull
git checkout mine
git rebase master

pip install -e . --verbose

```