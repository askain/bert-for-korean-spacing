import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import BertConfig, BertModel, AdamW
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score
from torchcrf import CRF
import numpy as np


from utils import load_slot_labels


class SpacingBertModel(pl.LightningModule):
    def __init__(
        self,
        config,
        ner_train_dataloader: DataLoader,
        ner_val_dataloader: DataLoader,
        ner_test_dataloader: DataLoader,
    ):
        super().__init__()
        self.config = config
        self.ner_train_dataloader = ner_train_dataloader
        self.ner_val_dataloader = ner_val_dataloader
        self.ner_test_dataloader = ner_test_dataloader
        self.slot_labels_type = load_slot_labels()
        self.pad_token_id = 0

        self.bert_config = BertConfig.from_pretrained(
            self.config.bert_model, num_labels=len(self.slot_labels_type)
        )
        self.model = BertModel.from_pretrained(
            self.config.bert_model, config=self.bert_config
        )
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.linear = nn.Linear(
            self.bert_config.hidden_size, len(self.slot_labels_type)
        )
        self.crf = CRF(len(self.slot_labels_type), batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_seq_out, _  = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        bert_seq_out = self.dropout(bert_seq_out)
        bert_features = self.linear(bert_seq_out)
        # return bert_features
        # sequence_of_tags = self.crf.decode(bert_features)
        sequence_of_tags = torch.from_numpy(np.array(self.crf.decode(bert_features)))
        self.temp_bert_features = bert_features
        return sequence_of_tags

    def training_step(self, batch, batch_nb):

        input_ids, attention_mask, token_type_ids, slot_labels = batch

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        loss = self._calculate_loss(outputs, slot_labels)
        tensorboard_logs = {"train_loss": loss}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):

        input_ids, attention_mask, token_type_ids, slot_labels = batch

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        loss = self._calculate_loss(outputs, slot_labels)
        gt_slot_labels, pred_slot_labels = self._convert_ids_to_labels(
            outputs, slot_labels
        )

        val_acc = self._f1_score(gt_slot_labels, pred_slot_labels)

        return {"val_loss": loss, "val_acc": val_acc}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        tensorboard_log = {
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        return {"val_loss": val_loss, "progress_bar": tensorboard_log}

    def test_step(self, batch, batch_nb):

        input_ids, attention_mask, token_type_ids, slot_labels = batch

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        gt_slot_labels, pred_slot_labels = self._convert_ids_to_labels(
            outputs, slot_labels
        )

        test_acc = self._f1_score(gt_slot_labels, pred_slot_labels)

        test_step_outputs = {
            "test_acc": test_acc,
            "gt_labels": gt_slot_labels,
            "pred_labels": pred_slot_labels,
        }

        return test_step_outputs

    def test_epoch_end(self, outputs):
        test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()

        gt_labels = []
        pred_labels = []
        for x in outputs:
            gt_labels.extend(x["gt_labels"])
            pred_labels.extend(x["pred_labels"])

        test_step_outputs = {
            "test_acc": test_acc,
            "gt_labels": gt_labels,
            "pred_labels": pred_labels,
        }

        return test_step_outputs

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)

    def train_dataloader(self):
        return self.ner_train_dataloader

    def val_dataloader(self):
        return self.ner_val_dataloader

    def test_dataloader(self):
        return self.ner_test_dataloader

    def _calculate_loss(self, outputs, labels):
        # active_logits = outputs.view(-1, len(self.slot_labels_type))
        # active_labels = labels.view(-1)
        # loss = F.cross_entropy(active_logits, active_labels)

        # active_logits = outputs.detach().cpu().numpy()
        # active_labels = labels.detach().cpu().numpy()
        # loss = torch.from_numpy(np.array((active_logits != active_labels).sum() / active_logits.size)).requires_grad_()
        
        # y_true = K.cast(K.argmax(y_true, axis=-1), dtype='int32')
        log_likelihood = self.crf(self.temp_bert_features, labels)
        del self.temp_bert_features
        return -log_likelihood
        loss = tf.reduce_mean(-log_likelihood)
        
        return loss
        
    def _f1_score(self, gt_slot_labels, pred_slot_labels):
        return torch.tensor(
            f1_score(gt_slot_labels, pred_slot_labels), dtype=torch.float32
        )

    def _convert_ids_to_labels(self, outputs, slot_labels):
        # _, y_hat = torch.max(outputs, dim=2)
        y_hat = outputs.detach().cpu().numpy()
        slot_label_ids = slot_labels.detach().cpu().numpy()

        slot_label_map = {i: label for i, label in enumerate(self.slot_labels_type)}
        slot_gt_labels = [[] for _ in range(slot_label_ids.shape[0])]
        slot_pred_labels = [[] for _ in range(slot_label_ids.shape[0])]

        for i in range(slot_label_ids.shape[0]):
            for j in range(slot_label_ids.shape[1]):
                if slot_label_ids[i, j] != self.pad_token_id:
                    slot_gt_labels[i].append(slot_label_map[slot_label_ids[i][j]])
                    slot_pred_labels[i].append(slot_label_map[y_hat[i][j]])

        return slot_gt_labels, slot_pred_labels


# class SimpleLSTM(nn.Module):
#     """
#     Simple LSTM model copied from:
#     https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
#     """

#     def __init__(self, vocab_size, nb_labels, emb_dim=10, hidden_dim=10):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.emb = nn.Embedding(vocab_size, emb_dim)
#         self.lstm = nn.LSTM(
#             emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True
#         )
#         self.hidden2tag = nn.Linear(hidden_dim, nb_labels)
#         self.hidden = None

#     def init_hidden(self, batch_size):
#         return (
#             torch.randn(2, batch_size, self.hidden_dim // 2),
#             torch.randn(2, batch_size, self.hidden_dim // 2),
#         )

#     def forward(self, batch_of_sentences):
#         self.hidden = self.init_hidden(batch_of_sentences.shape[0])
#         x = self.emb(batch_of_sentences)
#         x, self.hidden = self.lstm(x, self.hidden)
#         x = self.hidden2tag(x)
#         return x
        
# class BiLSTM_CRF(nn.Module):
#     def __init__(self, vocab_size, nb_labels, emb_dim=5, hidden_dim=4):
#         super().__init__()
#         self.lstm = SimpleLSTM(
#             vocab_size, nb_labels, emb_dim=emb_dim, hidden_dim=hidden_dim
#         )
#         self.crf = CRF(
#             nb_labels,
#             0,  # Const.BOS_TAG_ID,
#             0,  # Const.EOS_TAG_ID,
#             pad_tag_id=0,  # try setting pad_tag_id to None
#             batch_first=True,
#         )

#     def forward(self, x, mask=None):
#         emissions = self.lstm(x)
#         score, path = self.crf.decode(emissions, mask=mask)
#         return score, path

#     def loss(self, x, y, mask=None):
#         emissions = self.lstm(x)
#         nll = self.crf(emissions, y, mask=mask)
#         return nll

# class CRF(nn.Module):
#     """
#     Linear-chain Conditional Random Field (CRF).
#     Args:
#         nb_labels (int): number of labels in your tagset, including special symbols.
#         bos_tag_id (int): integer representing the beginning of sentence symbol in
#             your tagset.
#         eos_tag_id (int): integer representing the end of sentence symbol in your tagset.
#         pad_tag_id (int, optional): integer representing the pad symbol in your tagset.
#             If None, the model will treat the PAD as a normal tag. Otherwise, the model
#             will apply constraints for PAD transitions.
#         batch_first (bool): Whether the first dimension represents the batch dimension.
#     """

#     def __init__(
#         self, nb_labels, bos_tag_id, eos_tag_id, pad_tag_id=None, batch_first=True
#     ):
#         super().__init__()

#         self.nb_labels = nb_labels
#         self.BOS_TAG_ID = bos_tag_id
#         self.EOS_TAG_ID = eos_tag_id
#         self.PAD_TAG_ID = pad_tag_id
#         self.batch_first = batch_first

#         self.transitions = nn.Parameter(torch.empty(self.nb_labels, self.nb_labels))
#         self.init_weights()

#     def init_weights(self):
#         # initialize transitions from a random uniform distribution between -0.1 and 0.1
#         nn.init.uniform_(self.transitions, -0.1, 0.1)

#         # enforce contraints (rows=from, columns=to) with a big negative number
#         # so exp(-10000) will tend to zero

#         # no transitions allowed to the beginning of sentence
#         self.transitions.data[:, self.BOS_TAG_ID] = -10000.0
#         # no transition alloed from the end of sentence
#         self.transitions.data[self.EOS_TAG_ID, :] = -10000.0

#         if self.PAD_TAG_ID is not None:
#             # no transitions from padding
#             self.transitions.data[self.PAD_TAG_ID, :] = -10000.0
#             # no transitions to padding
#             self.transitions.data[:, self.PAD_TAG_ID] = -10000.0
#             # except if the end of sentence is reached
#             # or we are already in a pad position
#             self.transitions.data[self.PAD_TAG_ID, self.EOS_TAG_ID] = 0.0
#             self.transitions.data[self.PAD_TAG_ID, self.PAD_TAG_ID] = 0.0

#     def forward(self, emissions, tags, mask=None):
#         """Compute the negative log-likelihood. See `log_likelihood` method."""
#         nll = -self.log_likelihood(emissions, tags, mask=mask)
#         return nll

#     def log_likelihood(self, emissions, tags, mask=None):
#         """Compute the probability of a sequence of tags given a sequence of
#         emissions scores.
#         Args:
#             emissions (torch.Tensor): Sequence of emissions for each label.
#                 Shape of (batch_size, seq_len, nb_labels) if batch_first is True,
#                 (seq_len, batch_size, nb_labels) otherwise.
#             tags (torch.LongTensor): Sequence of labels.
#                 Shape of (batch_size, seq_len) if batch_first is True,
#                 (seq_len, batch_size) otherwise.
#             mask (torch.FloatTensor, optional): Tensor representing valid positions.
#                 If None, all positions are considered valid.
#                 Shape of (batch_size, seq_len) if batch_first is True,
#                 (seq_len, batch_size) otherwise.
#         Returns:
#             torch.Tensor: the (summed) log-likelihoods of each sequence in the batch.
#                 Shape of (1,)
#         """

#         # fix tensors order by setting batch as the first dimension
#         if not self.batch_first:
#             emissions = emissions.transpose(0, 1)
#             tags = tags.transpose(0, 1)

#         if mask is None:
#             mask = torch.ones(emissions.shape[:2], dtype=torch.float)

#         scores = self._compute_scores(emissions, tags, mask=mask)
#         partition = self._compute_log_partition(emissions, mask=mask)
#         return torch.sum(scores - partition)

#     def decode(self, emissions, mask=None):
#         """Find the most probable sequence of labels given the emissions using
#         the Viterbi algorithm.
#         Args:
#             emissions (torch.Tensor): Sequence of emissions for each label.
#                 Shape (batch_size, seq_len, nb_labels) if batch_first is True,
#                 (seq_len, batch_size, nb_labels) otherwise.
#             mask (torch.FloatTensor, optional): Tensor representing valid positions.
#                 If None, all positions are considered valid.
#                 Shape (batch_size, seq_len) if batch_first is True,
#                 (seq_len, batch_size) otherwise.
#         Returns:
#             torch.Tensor: the viterbi score for the for each batch.
#                 Shape of (batch_size,)
#             list of lists: the best viterbi sequence of labels for each batch.
#         """
#         if mask is None:
#             mask = torch.ones(emissions.shape[:2], dtype=torch.float)

#         scores, sequences = self._viterbi_decode(emissions, mask)
#         return scores, sequences

#     def _compute_scores(self, emissions, tags, mask):
#         """Compute the scores for a given batch of emissions with their tags.
#         Args:
#             emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
#             tags (Torch.LongTensor): (batch_size, seq_len)
#             mask (Torch.FloatTensor): (batch_size, seq_len)
#         Returns:
#             torch.Tensor: Scores for each batch.
#                 Shape of (batch_size,)
#         """
#         batch_size, seq_length = tags.shape
#         scores = torch.zeros(batch_size)

#         # save first and last tags to be used later
#         first_tags = tags[:, 0]
#         last_valid_idx = mask.int().sum(1) - 1
#         last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()

#         # add the transition from BOS to the first tags for each batch
#         t_scores = self.transitions[self.BOS_TAG_ID, first_tags]

#         # add the [unary] emission scores for the first tags for each batch
#         # for all batches, the first word, see the correspondent emissions
#         # for the first tags (which is a list of ids):
#         # emissions[:, 0, [tag_1, tag_2, ..., tag_nblabels]]
#         e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()

#         # the scores for a word is just the sum of both scores
#         scores += e_scores + t_scores

#         # now lets do this for each remaining word
#         for i in range(1, seq_length):

#             # we could: iterate over batches, check if we reached a mask symbol
#             # and stop the iteration, but vecotrizing is faster due to gpu,
#             # so instead we perform an element-wise multiplication
#             is_valid = mask[:, i]

#             previous_tags = tags[:, i - 1]
#             current_tags = tags[:, i]

#             # calculate emission and transition scores as we did before
#             e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
#             t_scores = self.transitions[previous_tags, current_tags]

#             # apply the mask
#             e_scores = e_scores * is_valid
#             t_scores = t_scores * is_valid

#             scores += e_scores + t_scores

#         # add the transition from the end tag to the EOS tag for each batch
#         scores += self.transitions[last_tags, self.EOS_TAG_ID]

#         return scores

#     def _compute_log_partition(self, emissions, mask):
#         """Compute the partition function in log-space using the forward-algorithm.
#         Args:
#             emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
#             mask (Torch.FloatTensor): (batch_size, seq_len)
#         Returns:
#             torch.Tensor: the partition scores for each batch.
#                 Shape of (batch_size,)
#         """
#         batch_size, seq_length, nb_labels = emissions.shape

#         # in the first iteration, BOS will have all the scores
#         alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]

#         for i in range(1, seq_length):
#             alpha_t = []

#             for tag in range(nb_labels):

#                 # get the emission for the current tag
#                 e_scores = emissions[:, i, tag]

#                 # broadcast emission to all labels
#                 # since it will be the same for all previous tags
#                 # (bs, nb_labels)
#                 e_scores = e_scores.unsqueeze(1)

#                 # transitions from something to our tag
#                 t_scores = self.transitions[:, tag]

#                 # broadcast the transition scores to all batches
#                 # (bs, nb_labels)
#                 t_scores = t_scores.unsqueeze(0)

#                 # combine current scores with previous alphas
#                 # since alphas are in log space (see logsumexp below),
#                 # we add them instead of multiplying
#                 scores = e_scores + t_scores + alphas

#                 # add the new alphas for the current tag
#                 alpha_t.append(torch.logsumexp(scores, dim=1))

#             # create a torch matrix from alpha_t
#             # (bs, nb_labels)
#             new_alphas = torch.stack(alpha_t).t()

#             # set alphas if the mask is valid, otherwise keep the current values
#             is_valid = mask[:, i].unsqueeze(-1)
#             alphas = is_valid * new_alphas + (1 - is_valid) * alphas

#         # add the scores for the final transition
#         last_transition = self.transitions[:, self.EOS_TAG_ID]
#         end_scores = alphas + last_transition.unsqueeze(0)

#         # return a *log* of sums of exps
#         return torch.logsumexp(end_scores, dim=1)

#     def _viterbi_decode(self, emissions, mask):
#         """Compute the viterbi algorithm to find the most probable sequence of labels
#         given a sequence of emissions.
#         Args:
#             emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
#             mask (Torch.FloatTensor): (batch_size, seq_len)
#         Returns:
#             torch.Tensor: the viterbi score for the for each batch.
#                 Shape of (batch_size,)
#             list of lists of ints: the best viterbi sequence of labels for each batch
#         """
#         batch_size, seq_length, nb_labels = emissions.shape

#         # in the first iteration, BOS will have all the scores and then, the max
#         alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]

#         backpointers = []

#         for i in range(1, seq_length):
#             alpha_t = []
#             backpointers_t = []

#             for tag in range(nb_labels):

#                 # get the emission for the current tag and broadcast to all labels
#                 e_scores = emissions[:, i, tag]
#                 e_scores = e_scores.unsqueeze(1)

#                 # transitions from something to our tag and broadcast to all batches
#                 t_scores = self.transitions[:, tag]
#                 t_scores = t_scores.unsqueeze(0)

#                 # combine current scores with previous alphas
#                 scores = e_scores + t_scores + alphas

#                 # so far is exactly like the forward algorithm,
#                 # but now, instead of calculating the logsumexp,
#                 # we will find the highest score and the tag associated with it
#                 max_score, max_score_tag = torch.max(scores, dim=-1)

#                 # add the max score for the current tag
#                 alpha_t.append(max_score)

#                 # add the max_score_tag for our list of backpointers
#                 backpointers_t.append(max_score_tag)

#             # create a torch matrix from alpha_t
#             # (bs, nb_labels)
#             new_alphas = torch.stack(alpha_t).t()

#             # set alphas if the mask is valid, otherwise keep the current values
#             is_valid = mask[:, i].unsqueeze(-1)
#             alphas = is_valid * new_alphas + (1 - is_valid) * alphas

#             # append the new backpointers
#             backpointers.append(backpointers_t)

#         # add the scores for the final transition
#         last_transition = self.transitions[:, self.EOS_TAG_ID]
#         end_scores = alphas + last_transition.unsqueeze(0)

#         # get the final most probable score and the final most probable tag
#         max_final_scores, max_final_tags = torch.max(end_scores, dim=1)

#         # find the best sequence of labels for each sample in the batch
#         best_sequences = []
#         emission_lengths = mask.int().sum(dim=1)
#         for i in range(batch_size):

#             # recover the original sentence length for the i-th sample in the batch
#             sample_length = emission_lengths[i].item()

#             # recover the max tag for the last timestep
#             sample_final_tag = max_final_tags[i].item()

#             # limit the backpointers until the last but one
#             # since the last corresponds to the sample_final_tag
#             sample_backpointers = backpointers[: sample_length - 1]

#             # follow the backpointers to build the sequence of labels
#             sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)

#             # add this path to the list of best sequences
#             best_sequences.append(sample_path)

#         return max_final_scores, best_sequences

#     def _find_best_path(self, sample_id, best_tag, backpointers):
#         """Auxiliary function to find the best path sequence for a specific sample.
#             Args:
#                 sample_id (int): sample index in the range [0, batch_size)
#                 best_tag (int): tag which maximizes the final score
#                 backpointers (list of lists of tensors): list of pointers with
#                 shape (seq_len_i-1, nb_labels, batch_size) where seq_len_i
#                 represents the length of the ith sample in the batch
#             Returns:
#                 list of ints: a list of tag indexes representing the bast path
#         """

#         # add the final best_tag to our best path
#         best_path = [best_tag]

#         # traverse the backpointers in backwards
#         for backpointers_t in reversed(backpointers):

#             # recover the best_tag at this timestep
#             best_tag = backpointers_t[best_tag][sample_id].item()

#             # append to the beginning of the list so we don't need to reverse it later
#             best_path.insert(0, best_tag)

#         return best_path