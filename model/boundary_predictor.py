#python main.py --mode train --data_path "/root/Morpheme-Aware MAGNET/data" --output_dir "/root/Morpheme-Aware MAGNET/output" --num_epochs 30 --batch_size 16 --learning_rate 1e-5 --lambda_boundary 0.0 --lambda_morph 0.05 --seed 42 --device cuda
 
"""BoundaryPrediction module for Morpheme-Aware MAGNET-style Models"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryPredictor(nn.Module):
    """
    Language-aware, morpheme-informed boundary predictor supporting Gumbel-Sigmoid and per-sample priors.
    """

    def __init__(self, d_model, d_inner, activation_function='gelu', temp=1.0, bp_type='gumbel', threshold=0.5):
        """
        Args:
            d_model: Size of hidden representations.
            d_inner: Hidden size in boundary MLP.
            activation_function: 'relu' or 'gelu'.
            temp: Initial Gumbel-Sigmoid temperature.
            bp_type: 'gumbel', 'entropy', or 'unigram'.
            threshold: Threshold for hard boundary.
        """
        super().__init__()
        self.temp = temp
        self.bp_type = bp_type
        self.threshold = threshold

        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = nn.GELU()
        else:
            raise ValueError("Unsupported activation!")

        self.boundary_predictor = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Linear(d_inner, 1)
        )

    def forward(self, hidden, temperature=None, return_logits_and_probs=False):
        """
        Args:
            hidden: [B, L, d_model]
            temperature: (Optional) override Gumbel temperature.
            return_logits_and_probs: If True, returns (logits, probs, hard_bins)
        Returns:
            boundary_probs: [B, L, 1]
        """
        logits = self.boundary_predictor(hidden)  # [B, L, 1]

        # Probabilities (always needed)
        probs = torch.sigmoid(logits)

        if self.bp_type == 'gumbel':
            temp = self.temp if temperature is None else temperature
            # Gumbel-Sigmoid noise for differentiable sampling
            uniform1 = torch.rand_like(logits)
            uniform2 = torch.rand_like(logits)
            gumbel1 = -torch.log(-torch.log(uniform1 + 1e-10) + 1e-10)
            gumbel2 = -torch.log(-torch.log(uniform2 + 1e-10) + 1e-10)
            noisy = (logits + gumbel1 - gumbel2) / temp
            soft_prob = torch.sigmoid(noisy)
            # Straight-through for hard discrete boundaries
            hard_boundaries = (soft_prob > self.threshold).float()
            hard_boundaries = hard_boundaries - soft_prob.detach() + soft_prob
        else:
            soft_prob = probs
            hard_boundaries = (soft_prob > self.threshold).float()

        if return_logits_and_probs:
            return logits, probs, hard_boundaries
        else:
            return soft_prob

    def calc_binomial_count_loss(self, probs, priors, attention_mask):
        """
        Binomial negative log-likelihood for soft boundary count matching.
        Args:
            probs: Boundary probabilities, [B, L], after sigmoid.
            priors: Prior probability for each sample, [B] (e.g., 0.12 for Hindi).
            attention_mask: Mask for valid positions, [B, L].
        Returns:
            Scalar boundary count loss.
        """
        # [B, L] -> sum over sequence (valid only)
        valid_probs = probs * attention_mask  # [B, L]
        sum_preds = valid_probs.sum(dim=1)    # [B]
        total_count = attention_mask.sum(dim=1).float()  # [B]

        # Clamp priors and counts for safety
        priors = torch.clamp(priors, 1e-6, 1-1e-6)
        total_count = total_count.clamp(min=1)

        # [B] Binomial NLL
        binomial = torch.distributions.binomial.Binomial(total_count, probs=priors)
        count_loss = -binomial.log_prob(sum_preds).clamp(max=1e4).mean()
        return count_loss

    def calc_supervised_bce_loss(self, logits, gold_boundaries, attention_mask):
        """
        Supervised BCE (token-level) boundary loss, masked.
        Args:
            logits: [B, L, 1] or [B, L]
            gold_boundaries: [B, L]
            attention_mask: [B, L]
        Returns:
            Scalar supervised boundary loss.
        """
        if logits.dim() == 3:
            logits = logits.squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, gold_boundaries.float(), reduction='none')
        loss = (loss * attention_mask).sum() / attention_mask.sum().clamp(min=1)
        return loss

    def calc_fuzzy_supervised_loss(self, logits, gold_boundaries, attention_mask, window=2):
        """
        Fuzzy window BCE: Accepts boundaries close to gold within a window.
        Args:
            logits: [B, L, 1] or [B, L]
            gold_boundaries: [B, L]
            attention_mask: [B, L]
            window: Window size for fuzzy matching.
        Returns:
            Scalar fuzzy boundary loss.
        """
        if logits.dim() == 3:
            logits = logits.squeeze(-1)
        # Construct soft targets: max over window (per position)
        max_targets = gold_boundaries
        for shift in range(-window, window+1):
            shifted = torch.roll(gold_boundaries, shifts=shift, dims=1)
            max_targets = torch.max(max_targets, shifted)
        loss = F.binary_cross_entropy_with_logits(logits, max_targets.float(), reduction='none')
        loss = (loss * attention_mask).sum() / attention_mask.sum().clamp(min=1)
        return loss

    def calc_entropy_loss(self, probs, attention_mask):
        """
        Entropy regularization for boundary probabilities.
        Args:
            probs: [B, L]
            attention_mask: [B, L]
        Returns: scalar entropy loss.
        """
        eps = 1e-8
        ent = -(probs * (probs + eps).log() + (1 - probs)*(1 - probs + eps).log())
        ent = (ent * attention_mask).sum() / attention_mask.sum().clamp(min=1)
        return ent

    def calc_stats(self, preds, gt):
        # preds/gt: [B, L]
        preds = preds.bool()
        gt = gt.bool()
        TP = ((preds == gt) & preds).sum().item()
        FP = ((preds != gt) & preds).sum().item()
        FN = ((preds != gt) & (~preds)).sum().item()
        acc = (preds == gt).sum().item() / gt.numel() if gt.numel() > 0 else 0
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
        return {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}


# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class BoundaryPredictor(nn.Module):
#     def __init__(self, d_model, d_inner, activation_function,
#                  temp, prior, bp_type, threshold=0.5):
#         super().__init__()
#         self.temp = temp
#         self.prior = prior
#         self.bp_type = bp_type
#         self.threshold = threshold

#         if activation_function == 'relu':
#             activation_fn = nn.ReLU(inplace=True)
#         elif activation_function == 'gelu':
#             activation_fn = torch.nn.GELU()

#         self.boundary_predictor = nn.Sequential(
#             nn.Linear(d_model, d_inner),
#             activation_fn,
#             nn.Linear(d_inner, 1),
#         )

#         self.loss = nn.BCEWithLogitsLoss()

#     def forward(self, hidden):
#         # Hidden is of shape [batch_size, seq_len, d_model]
#         # Boundaries we return are [batch_size, seq_len]

#         boundary_logits = self.boundary_predictor(hidden)
#         boundary_probs = torch.sigmoid(boundary_logits)

#         if self.bp_type == 'gumbel':
#             bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
#                 temperature=self.temp,
#                 probs=boundary_probs,
#             )

#             soft_boundaries = bernoulli.rsample()

#             hard_boundaries = (soft_boundaries > self.threshold).float()
#             hard_boundaries = (
#                 hard_boundaries - soft_boundaries.detach() + soft_boundaries
#             )
#         elif self.bp_type in ['entropy', 'unigram']:
#             soft_boundaries = boundary_probs
#             hard_boundaries = (soft_boundaries > self.threshold).float()

#         return soft_boundaries, hard_boundaries

#     def calc_loss_without_padding(self, preds, gt, attention_mask):
#         """
#         Calculate boundary loss, accounting for padding tokens.
#         """
#         # B x T
#         if self.bp_type in ['entropy', 'unigram']:
#             assert preds is not None and gt is not None
#             return self.loss(preds, gt.float())

#         elif self.bp_type in ['gumbel']:
#             assert attention_mask is not None and gt is None

#             # create a mask based on attention_mask
#             mask = attention_mask.eq(1)  # Mask is True where tokens are present, False for padding

#             # apply the mask to predictions
#             masked_preds = preds * mask.float()

#             # Compute the sum of predictions for each example in the batch
#             sum_preds = masked_preds.sum(dim=-1).unsqueeze(dim=-1)

#             # Compute the total count of trials for each example in the batch
#             total_count = mask.sum(dim=-1, keepdim=True).float()  # Number of non-padded tokens

#             # compute the sum of predictions for each example in the batch
#             binomial = torch.distributions.binomial.Binomial(
#                     total_count,
#                     probs=torch.Tensor([self.prior]).to(preds.device)
#                 )
#             loss_boundaries = -binomial.log_prob(
#                     sum_preds
#                 ).mean()

#             return loss_boundaries

#     def calc_loss(self, preds, gt):
#         # B x T
#         if self.bp_type in ['entropy', 'unigram']:
#             assert preds is not None and gt is not None
#             return self.loss(preds, gt.float())
#         elif self.bp_type in ['gumbel']:
#             assert gt is None
#             binomial = torch.distributions.binomial.Binomial(
#                 preds.size(-1),
#                 probs=torch.Tensor([self.prior]).to(preds.device)
#             )
#             loss_boundaries = -binomial.log_prob(
#                 preds.sum(dim=-1)
#             ).mean() / preds.size(-1)

#             return loss_boundaries

#     def calc_stats(self, preds, gt):
#         # B x T
#         preds, gt = preds.bool(), gt.bool()
#         TP = ((preds == gt) & preds).sum().item()
#         FP = ((preds != gt) & preds).sum().item()
#         FN = ((preds != gt) & (~preds)).sum().item()

#         acc = (preds == gt).sum().item() / gt.numel()

#         if TP == 0:
#             precision, recall = 0, 0
#         else:
#             precision = TP / (TP + FP)
#             recall = TP / (TP + FN)

#         stats = {
#             'acc': acc,
#             'precision': precision,
#             'recall': recall
#         }

#         return stats