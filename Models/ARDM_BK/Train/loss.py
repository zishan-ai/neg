import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self):
         super().__init__()

    def forward(self, logits, targets, mask, label_smoothing=-1, reduce=None):
        return self.sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce)

    def sequence_cross_entropy_with_logits(self, logits, targets, mask, label_smoothing, reduce):
        # logits ---> (1, sequences, num_classes)  eg: torch.Size([1, 620, 50257])

        # shape: (batch * sequence_length, num_classes) eg: torch.Size([620, 50257])
        logits_flat = logits.view(-1, logits.size(-1))

        # shape: (batch * sequence_length, num_classes) eg: torch.Size([620, 50257]) 
        log_probs_flat = F.log_softmax(logits_flat, dim=-1)

        # targets --> torch.Size([1, 620])
        # shape: (batch * max_len, 1) eg: torch.Size([620, 1])
        targets_flat = targets.view(-1, 1).long()

        if label_smoothing > 0.0:
            num_classes = logits.size(-1)
            smoothing_value = label_smoothing / float(num_classes)
            # Fill all the correct indices with 1 - smoothing value.
            one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
            smoothed_targets = one_hot_targets + smoothing_value
            # smoothed_targets eg: torch.Size([620, 50257])
            negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
            negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
        else:
            # shape: (batch * sequence_length, 1)
            negative_log_likelihood_flat = - torch.gather(log_probs_flat.to('cuda'), dim=1, index=targets_flat.to('cuda'))

        # shape: (1, batch * sequence_length)
        negative_log_likelihood = negative_log_likelihood_flat.view(-1, logits.shape[1])

        # shape : (batch, sequence_length)
        loss = negative_log_likelihood.to('cuda') * mask.to('cuda')

        if reduce:
            # shape: (batch,)
            loss = loss.sum(1) / (mask.sum(1) + 1e-13)

            if reduce is "batch":
                loss = loss.mean()

        return loss