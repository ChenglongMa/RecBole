# TODO

[ ] Add `valid_eval_args` and `test_eval_args`;

[ ] Change `eval_neg_sample_args` to `valid_neg_sample_args` and `test_neg_sample_args`;

1). used_item refers to all items that users have interacted in the training and evaluation set. positve_item refers to all items that users have interacted in the evaluation set. history_item refers to all items that users have interacted in the training set.

2). positive_u and positive_i are used to calculate evaluation scores. positive_u(Torch.Tensor) should be the row index of positive items for each user in the evaluation set, and positive_i(Torch.Tensor) should be the positive item id for each user.