# @Time   : 2020/10/19
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2021/7/9
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""
recbole.data.customized_dataset
##################################

We only recommend building customized datasets by inheriting.

Customized datasets named ``[Model Name]Dataset`` can be automatically called.
"""
from collections import defaultdict

import numpy as np
import torch

from recbole.data.dataset import KGSeqDataset, SequentialDataset
from recbole.data.interaction import Interaction
from recbole.sampler import SeqSampler, MaskedSeqSampler
from recbole.utils.enum_type import FeatureType, FeatureSource


class GRU4RecKGDataset(KGSeqDataset):
    def __init__(self, config):
        super().__init__(config)


class KSRDataset(KGSeqDataset):
    def __init__(self, config):
        super().__init__(config)


class DIENDataset(SequentialDataset):
    """:class:`DIENDataset` is based on :class:`~recbole.data.dataset.sequential_dataset.SequentialDataset`.
    It is different from :class:`SequentialDataset` in `data_augmentation`.
    It add users' negative item list to interaction.

    The original version of sampling negative item list is implemented by Zhichao Feng (fzcbupt@gmail.com) in 2021/2/25,
    and he updated the codes in 2021/3/19. In 2021/7/9, Yupeng refactored SequentialDataset & SequentialDataLoader,
    then refactored DIENDataset, either.

    Attributes:
        augmentation (bool): Whether the interactions should be augmented in RecBole.
        seq_sample (recbole.sampler.SeqSampler): A sampler used to sample negative item sequence.
        neg_item_list_field (str): Field name for negative item sequence.
        neg_item_list (torch.tensor): all users' negative item history sequence.
    """

    def __init__(self, config):
        super().__init__(config)

        list_suffix = config["LIST_SUFFIX"]
        neg_prefix = config["NEG_PREFIX"]
        self.seq_sampler = SeqSampler(self)
        self.neg_item_list_field = neg_prefix + self.iid_field + list_suffix
        self.neg_item_list = self.seq_sampler.sample_neg_sequence(
            self.inter_feat[self.iid_field]
        )

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``
        """
        self.logger.debug("data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                if (
                        self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                        and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                # DIEN
                list_ftype = self.field2type[list_field]
                dtype = (
                    torch.int64
                    if list_ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]
                    else torch.float64
                )
                # End DIEN
                new_dict[list_field] = torch.zeros(shape, dtype=dtype)

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(
                        zip(item_list_index, item_list_length)
                ):
                    new_dict[list_field][i][:length] = value[index]

                # DIEN
                if field == self.iid_field:
                    new_dict[self.neg_item_list_field] = torch.zeros(shape, dtype=dtype)
                    for i, (index, length) in enumerate(
                            zip(item_list_index, item_list_length)
                    ):
                        new_dict[self.neg_item_list_field][i][
                        :length
                        ] = self.neg_item_list[index]
                # End DIEN

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data


class MaskedSequentialDataset(SequentialDataset):
    """:class:`MaskedSequentialDataset` is based on :class:`~recbole.data.dataset.sequential_dataset.SequentialDataset`.
    It is different from :class:`SequentialDataset` in `data_augmentation`.
    It adds users' negative item list to interaction.

    The original version of sampling negative item list is implemented by Zhichao Feng (fzcbupt@gmail.com) in 2021/2/25,
    and he updated the codes in 2021/3/19. In 2021/7/9, Yupeng refactored SequentialDataset & SequentialDataLoader,
    then refactored MaskedSequentialDataset, either.

    Attributes:
        augmentation (bool): Whether the interactions should be augmented in RecBole.
        seq_sample (recbole.sampler.SeqSampler): A sampler used to sample negative item sequence.
        neg_item_list_field (str): Field name for negative item sequence.
        neg_item_list (torch.tensor): all users' negative item history sequence.
    """

    def _get_field_from_config(self):
        super()._get_field_from_config()
        list_suffix = self.config["LIST_SUFFIX"]
        neg_prefix = self.config["NEG_PREFIX"]
        self.neg_item_list_field = neg_prefix + self.iid_field + list_suffix  # default: neg_item_list
        self.mask_field = self.config['MASK_FIELD']

    def _benchmark_presets(self):
        list_suffix = self.config["LIST_SUFFIX"]
        for field in self.inter_feat:
            if field + list_suffix in self.inter_feat:
                list_field = field + list_suffix
                setattr(self, f"{field}_list_field", list_field)
        self.set_field_property(self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)

        if hasattr(self, 'item_id_list_field'):
            self.inter_feat[self.item_list_length_field] = self.inter_feat[self.item_id_list_field].agg(len)
        else:
            for feat_name in self.feat_name_list:
                feat = getattr(self, feat_name)
                setattr(self, feat_name, self._dataframe_to_interaction(feat))
            self.data_augmentation()

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``

        refer to https://github.com/RUCAIBox/RecBole/issues/1787#issuecomment-1575977154

        """
        self.logger.debug("data_augmentation")

        # mcl: added
        # self.seq_sampler = SeqSampler(self)
        seq_sampler = MaskedSeqSampler(self, distribution="uniform", alpha=1.0)
        neg_item_list, neg_item_masks = seq_sampler.sample_neg_sequence(self.inter_feat[self.iid_field].numpy())

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        # uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                if (
                        self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                        and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                # TICEN
                list_ftype = self.field2type[list_field]
                dtype = (
                    torch.int64
                    if list_ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]
                    else torch.float64
                )
                # End TICEN
                new_dict[list_field] = torch.zeros(shape, dtype=dtype)

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(
                        zip(item_list_index, item_list_length)
                ):
                    new_dict[list_field][i][:length] = value[index]

                # TICEN
                if field == self.iid_field:
                    new_dict[self.neg_item_list_field] = torch.zeros(shape, dtype=dtype)
                    new_dict[self.mask_field] = torch.zeros(shape, dtype=torch.bool)
                    for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                        new_dict[self.neg_item_list_field][i][:length] = neg_item_list[index]
                        new_dict[self.mask_field][i][:length] = neg_item_masks[index]
                # End TICEN

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
                See :class:`~recbole.config.eval_setting.EvalSetting` for details.

                Args:
                    eval_setting (:class:`~recbole.config.eval_setting.EvalSetting`):
                        Object contains evaluation settings, which guide the data processing procedure.

                Returns:
                    list: List of built :class:`Dataset`.
        """
        if self.benchmark_filename_list is not None:
            self._drop_unused_col()
            phase_field = self.config['PHASE_FIELD']
            if phase_field is None:
                cumsum = list(np.cumsum(self.file_size_list))
                datasets = [
                    self.copy(self.inter_feat[start:end])
                    for start, end in zip([0] + cumsum[:-1], cumsum)
                ]
                return datasets
            phase_index = defaultdict(list)
            phase_train = self.config['PHASE_TRAIN'] or 'train'
            phase_valid = self.config['PHASE_VALID'] or 'valid'
            phase_test = self.config['PHASE_TEST'] or 'test'

            for i, phase in enumerate(self.inter_feat[phase_field].numpy()):
                phase_index[self.field2id_token[phase_field][phase]].append(i)

            datasets = [self.copy(self.inter_feat[phase_index[key]]) for key in [phase_train, phase_valid, phase_test]]
            return datasets

        ordering_args = self.config["eval_args"]["order"]
        if ordering_args != "TO":
            raise ValueError(
                f"The ordering args for sequential recommendation has to be 'TO'"
            )

        return super().build()
