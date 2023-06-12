# -*- coding: utf-8 -*-
# @Time     : 2023/06/05 14:56
# @Author   : Chenglong Ma
# @Email    : chenglong.m@outlook.com
import numpy as np
import torch
import torch.nn as nn
from recbole.model.layers import TransformerEncoder

from recbole.data.dataset.sequential_dataset import SequentialDataset
from torch.nn import TransformerEncoderLayer
from torch.nn.init import xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class Ours(SequentialRecommender):
    def __init__(self, config, dataset: SequentialDataset):
        super().__init__(config, dataset)

        # load the dataset information
        self.n_user = dataset.num(self.USER_ID)
        self.device = config["device"]
        self.TIME_SEQ = dataset.timestamp_list_field

        # load the parameters information
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.embedding_size = config["embedding_size"]
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.position_emb_scale = config["position_emb_scale"]

        # define layers and loss type
        # self.user_embedding = nn.Embedding(self.n_user, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        # self.embedding_seq_item = nn.Embedding(
        #     self.n_items, self.embedding_size, padding_idx=0
        # )

        # new layers
        self.max_position_length = self.position_ids(dataset.time_diff)
        self.position_embedding = nn.Embedding(self.max_position_length, self.embedding_size)

        # self.attention_layer = TransformerEncoderLayer(d_model=self.embedding_size, nhead=8, batch_first=True)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.embedding_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)

        # end of new
        # self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.loss_type = config["loss_type"]
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # init the parameters of the module
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def position_ids(self, time_diff):
        ids = np.ceil(self.position_emb_scale * np.log(time_diff + 1))
        if isinstance(time_diff, torch.Tensor):
            return ids.long()
        return ids.astype(int)

    def forward(self, item_seq, user, item_seq_len, time_seq=None):
        # user_embedding = self.user_embedding(user)
        extended_attention_mask = self.get_attention_mask(item_seq)

        seq_item_embedding = self.item_embedding(item_seq)
        time_diffs = (time_seq.max(dim=1, keepdim=True)[0] - time_seq).cpu()
        position_ids = self.position_ids(time_diffs).to(time_seq.device)
        position_ids = position_ids.clip(min=0, max=self.max_position_length - 1)
        position_embedding = self.position_embedding(position_ids)

        seq_item_embedding = seq_item_embedding + position_embedding

        seq_item_embedding = self.LayerNorm(seq_item_embedding)
        seq_item_embedding = self.dropout(seq_item_embedding)

        trm_output = self.trm_encoder(
            seq_item_embedding, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output
        # return user_embedding + output

    def get_seq_output(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        user = interaction[self.USER_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_seq = interaction[self.TIME_SEQ]
        seq_output = self.forward(item_seq, user, item_seq_len, time_seq)
        return seq_output

    def calculate_loss(self, interaction):
        seq_output = self.get_seq_output(interaction)

        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            pos_items_emb = self.item_embedding(pos_items)
            neg_items = interaction[self.NEG_ITEM_ID]
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        seq_output = self.get_seq_output(interaction)
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        seq_output = self.get_seq_output(interaction)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
