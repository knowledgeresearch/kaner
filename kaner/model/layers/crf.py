# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Conditional Random Fields"""

from typing import List

import torch
import torch.nn as nn


class CRF(nn.Module):
    """
    Compute the log-likelihood of the input assuming a conditional random field model.

    References:
        [1] https://people.cs.umass.edu/~wallach/technical_reports/wallach04conditional.pdf
        [2] https://nlp.cs.nyu.edu/nycnlp/lafferty01conditional.pdf
        [3] https://zhuanlan.zhihu.com/p/97676647
        [4] https://gist.github.com/koyo922/9300e5afbec83cbb63ad104d6a224cf4#file-bilstm_crf-py-L18
        [5] https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
            #bi-lstm-conditional-random-field-discussion

    Args:
        n_tags (int): The number of tags.
    """

    def __init__(self, n_tags: int, ignore_index: int, default_label_pad_index: int):
        assert isinstance(n_tags, int) and n_tags > 0
        super(CRF, self).__init__()
        self.n_tags = n_tags
        # Add two states at the end to accommodate start and end states
        # (i,j) element represents the probability of transitioning from state i to j
        self.transitions = nn.Parameter(torch.Tensor(n_tags + 2, n_tags + 2))
        self.start_tag = n_tags
        self.end_tag = n_tags + 1
        self.reset_parameters()
        self.ignore_index = ignore_index
        self.default_label_pad_index = default_label_pad_index

    def reset_parameters(self):
        """
        Initialize all parameters in CRF.
        """
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        self.transitions.data[:, self.start_tag] = -10000
        self.transitions.data[self.end_tag, :] = -10000

    def forward(self, emissions, mask, tags=None, reduce=True):
        """
        Compute log-likelihood of input or decoding.

        Args:
            emissions (torch.FloatTensor): Emission values for different tags for each input. The expected shape is
                (batch_size, seq_len, n_tags). Padding is should be on the right side of the input.
            mask (torch.FloatTensor): Batch sequence length with shape (batch_size, seq_len).
            tags (torch.LongTensor): Actual tags for each token in the input. Expected shape is (batch_size, seq_len).
        """
        if tags is not None:
            numerator = self._compute_joint_llh(emissions, tags, mask)
            denominator = self._compute_log_partition_function(emissions, mask)
            llh = numerator - denominator
            loss = -(llh if not reduce else torch.mean(llh))

            return loss.unsqueeze(0)
        # decoding
        result = self._viterbi_decode(emissions, mask)

        return result

    def _compute_joint_llh(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor):
        """
        Given a set of emission scores and the ground-truth tags, return its score.

        Args:
            emissions: Emission scores with expected shape of
                batch_size * seq_len * num_labels
            tags: Actual tags for each token in the input. Expected shape is
                batch_size * seq_len
            mask: Mask for ignored tags, such as PAD. Expected shape is
                batch_size * seq_len
        """
        seq_len = emissions.shape[1]

        llh = self.transitions[self.start_tag, tags[:, 0]].unsqueeze(1)
        llh += emissions[:, 0, :].gather(1, tags[:, 0].view(-1, 1)) * mask[:, 0].unsqueeze(1)

        for idx in range(1, seq_len):
            old_state, new_state = (
                tags[:, idx - 1].view(-1, 1),
                tags[:, idx].view(-1, 1),
            )
            emission_scores = emissions[:, idx, :].gather(1, new_state)
            transition_scores = self.transitions[old_state, new_state]
            llh += (emission_scores + transition_scores) * mask[:, idx].unsqueeze(1)

        # Index of the last tag is calculated by taking the sum of mask matrix
        # for each input row and subtracting 1 from the sum.
        last_tag_indices = mask.sum(1, dtype=torch.long) - 1
        last_tags = tags.gather(1, last_tag_indices.view(-1, 1))

        llh += self.transitions[last_tags.squeeze(1), self.end_tag].unsqueeze(1)

        return llh.squeeze(1)

    def _compute_log_partition_function(self, emissions: torch.Tensor, mask: torch.Tensor):
        """
        Given a set of emission scores and the mask, return the logsum of all path's score.

        Args:
            emissions (torch.FloatTensor): Emission scores with expected shape of (batch_size, seq_len, n_tags).
            mask (torch.FloatTensor): Mask for ignored tags, such as PAD. Expected shape is (batch_size, seq_len).
        """
        seq_len = emissions.shape[1]

        log_prob = emissions[:, 0].clone()
        log_prob += self.transitions[self.start_tag, : self.start_tag].unsqueeze(0)

        for idx in range(1, seq_len):
            broadcast_emissions = emissions[:, idx].unsqueeze(1)
            broadcast_transitions = self.transitions[
                : self.start_tag, : self.start_tag
            ].unsqueeze(0)
            broadcast_logprob = log_prob.unsqueeze(2)
            score = broadcast_logprob + broadcast_emissions + broadcast_transitions

            score = torch.logsumexp(score, 1)
            log_prob = score * mask[:, idx].unsqueeze(1) + log_prob.squeeze(1) * (
                1 - mask[:, idx].unsqueeze(1)
            )

        log_prob += self.transitions[: self.start_tag, self.end_tag].unsqueeze(0)
        return torch.logsumexp(log_prob.squeeze(1), 1)

    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor):
        """
        Decode the tag path with maximum probability by Viterbi algorithm.

        Args:
            emissions (torch.FloatTensor): Emission scores with expected shape of (batch_size, seq_len, n_tags).
            mask (torch.FloatTensor): Mask for ignored tags, such as PAD. Expected shape is (batch_size, seq_len).
        """
        tensor_device = emissions.device
        seq_len = emissions.shape[1]
        mask = mask.to(torch.uint8)

        log_prob = emissions[:, 0].clone()
        log_prob += self.transitions[self.start_tag, : self.start_tag].unsqueeze(0)

        # At each step, we need to keep track of the total score, as if this step
        # was the last valid step.
        end_scores = log_prob + self.transitions[
            : self.start_tag, self.end_tag
        ].unsqueeze(0)

        best_scores_list: List[torch.Tensor] = []
        # Needed for Torchscript as empty list is assumed to be list of tensors
        empty_data: List[int] = []
        # If the element has only token, empty tensor in best_paths helps
        # torch.cat() from crashing
        best_paths_list = [torch.tensor(empty_data, device=tensor_device).long()]
        best_scores_list.append(end_scores.unsqueeze(1))

        for idx in range(1, seq_len):
            broadcast_emissions = emissions[:, idx].unsqueeze(1)
            broadcast_transmissions = self.transitions[
                : self.start_tag, : self.start_tag
            ].unsqueeze(0)
            broadcast_log_prob = log_prob.unsqueeze(2)

            score = broadcast_emissions + broadcast_transmissions + broadcast_log_prob

            max_scores, max_score_indices = torch.max(score, 1)

            best_paths_list.append(max_score_indices.unsqueeze(1))

            # Storing the scores incase this was the last step.
            end_scores = max_scores + self.transitions[
                : self.start_tag, self.end_tag
            ].unsqueeze(0)

            best_scores_list.append(end_scores.unsqueeze(1))
            log_prob = max_scores

        best_scores = torch.cat(best_scores_list, 1).float()
        best_paths = torch.cat(best_paths_list, 1)

        _, max_indices_from_scores = torch.max(best_scores, 2)

        valid_index_tensor = torch.tensor(0, device=tensor_device).long()
        if self.ignore_index == self.default_label_pad_index:
            # No label for padding, so use 0 index.
            padding_tensor = valid_index_tensor
        else:
            padding_tensor = torch.tensor(
                self.ignore_index, device=tensor_device
            ).long()

        # Label for the last position is always based on the index with max score
        # For illegal timesteps, we set as ignore_index
        labels = max_indices_from_scores[:, seq_len - 1]
        labels = self._mask_tensor(labels, 1 - mask[:, seq_len - 1], padding_tensor)

        all_labels = labels.unsqueeze(1).long()

        # For Viterbi decoding, we start at the last position and go towards first
        for idx in range(seq_len - 2, -1, -1):
            # There are two ways to obtain labels for tokens at a particular position.

            # Option 1: Use the labels obtained from the previous position to index
            # the path in present position. This is used for all positions except
            # last position in the sequence.
            # Option 2: Find the indices with maximum scores obtained during
            # viterbi decoding. This is used for the token at the last position

            # For option 1 need to convert invalid indices to 0 so that lookups
            # dont fail.
            indices_for_lookup = all_labels[:, -1].clone()
            indices_for_lookup = self._mask_tensor(
                indices_for_lookup,
                indices_for_lookup == self.ignore_index,
                valid_index_tensor,
            )

            # Option 1 is used here when previous timestep (idx+1) was valid.
            indices_from_prev_pos = (
                best_paths[:, idx, :]
                .gather(1, indices_for_lookup.view(-1, 1).long())
                .squeeze(1)
            )
            indices_from_prev_pos = self._mask_tensor(
                indices_from_prev_pos, (1 - mask[:, idx + 1]), padding_tensor
            )

            # Option 2 is used when last timestep was not valid which means idx+1
            # is the last position in the sequence.
            indices_from_max_scores = max_indices_from_scores[:, idx]
            indices_from_max_scores = self._mask_tensor(
                indices_from_max_scores, mask[:, idx + 1], padding_tensor
            )

            # We need to combine results from 1 and 2 as rows in a batch can have
            # sequences of varying lengths
            labels = torch.where(
                indices_from_max_scores == self.ignore_index,
                indices_from_prev_pos,
                indices_from_max_scores,
            )

            # Set to ignore_index if present state is not valid.
            labels = self._mask_tensor(labels, (1 - mask[:, idx]), padding_tensor)
            all_labels = torch.cat((all_labels, labels.view(-1, 1).long()), 1)

        return torch.flip(all_labels, [1])

    def _mask_tensor(self, score_tensor, mask_condition, mask_value):
        masked_tensor = torch.where(mask_condition, mask_value, score_tensor)
        return masked_tensor
