# loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_frame=1.0, cost_coord=1.0):
        """
        Initializes the HungarianMatcher.

        Args:
            cost_class (float): Weight for classification cost.
            cost_frame (float): Weight for frame index cost.
            cost_coord (float): Weight for coordinate cost.
        """
        super(HungarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_frame = cost_frame
        self.cost_coord = cost_coord

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Performs the matching between predictions and targets.

        Args:
            outputs (dict): Contains 'logits', 'frames', 'xy' tensors.
                - logits: (batch_size, num_queries, num_classes)
                - frames: (batch_size, num_queries)
                - xy: (batch_size, num_queries, 2)
            targets (list of dict): Each dict contains:
                - 'labels': (num_events,)
                - 'frames': (num_events,)
                - 'xy': (num_events, 2)
                - 'event_mask': (num_events,)

        Returns:
            List of matched indices for each batch element.
        """
        bs, num_queries, num_classes = outputs['logits'].shape
        indices = []

        for b in range(bs):
            # Extract predictions for the current batch element
            logits = outputs['logits'][b]    # (num_queries, num_classes)
            frames = outputs['frames'][b]    # (num_queries,)
            xy = outputs['xy'][b]            # (num_queries, 2)

            # Extract targets for the current batch element
            target = targets[b]
            tgt_labels = target['labels']    # (num_events,)
            tgt_frames = target['frames']    # (num_events,)
            tgt_xy = target['xy']            # (num_events, 2)

            num_targets = tgt_labels.shape[0]
            if num_targets == 0:
                raise ValueError(f"Batch element {b} has no valid targets.")

            # Compute cost matrix for the current batch element
            # Classification cost: Negative log-probability of the target class
            prob = F.softmax(logits, dim=-1)  # (num_queries, num_classes)
            # Gather the probabilities of the target classes
            # This requires tgt_labels to be within [0, num_classes-1]
            if torch.any(tgt_labels >= num_classes) or torch.any(tgt_labels < 0):
                raise ValueError(f"Target labels must be in [0, {num_classes-1}]")

            # cost_class: (num_queries, num_targets)
            cost_class = -prob[:, tgt_labels]  # Negative log-probability

            # Frame index cost: L1 distance between predicted and target frame indices
            # frames: (num_queries,), tgt_frames: (num_targets,)
            cost_frame = torch.abs(frames.unsqueeze(1) - tgt_frames.unsqueeze(0))  # (num_queries, num_targets)

            # Coordinate cost: L1 distance between predicted and target xy coordinates
            # xy: (num_queries, 2), tgt_xy: (num_targets, 2)
            cost_coord = torch.abs(xy.unsqueeze(1) - tgt_xy.unsqueeze(0)).sum(-1)  # (num_queries, num_targets)

            # Total cost
            cost_matrix = self.cost_class * cost_class + self.cost_frame * cost_frame + self.cost_coord * cost_coord  # (num_queries, num_targets)

            # Convert cost_matrix to numpy for scipy
            cost_matrix = cost_matrix.detach().cpu().numpy()

            # Perform Hungarian matching (linear_sum_assignment)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64, device=logits.device),
                torch.as_tensor(col_ind, dtype=torch.int64, device=logits.device)
            ))

        return indices


class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, eos_coef, num_classes, losses=["labels", "frames", "xy"]):
        """
        Create the criterion.

        Args:
            matcher (HungarianMatcher): Module able to compute a matching between targets and proposals.
            weight_dict (dict): Dict containing as key the names of the losses and as values their relative weight.
            eos_coef (float): Relative classification weight applied to the no-object class.
            num_classes (int): Number of classes (including background).
            losses (list of str): List of all the losses to be applied.
        """
        super(SetCriterion, self).__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        # Define the weight for the background class (0) and other classes
        empty_weight = torch.ones(num_classes, dtype=torch.float)
        empty_weight[0] = eos_coef  # Background class has lower weight
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_targets):
        """
        Classification loss (CrossEntropy)
        targets dicts must contain a "labels" key.
        """
        src_logits = outputs['logits']  # (batch_size, num_queries, num_classes)

        # Flatten to (batch_size*num_queries, num_classes)
        src_logits = src_logits.view(-1, src_logits.shape[-1])

        # Create target tensor filled with background class (0)
        target_classes = torch.full((src_logits.shape[0],), 0, dtype=torch.int64, device=src_logits.device)

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            # Assign target classes to the matched queries
            # src_idx: matched query indices
            # tgt_idx: matched target event indices
            target_classes[batch_idx * outputs['logits'].shape[1] + src_idx] = targets[batch_idx]['labels'][tgt_idx]

        # Compute cross-entropy loss with class weights
        loss_ce = F.cross_entropy(src_logits, target_classes, weight=self.empty_weight)

        return {'loss_ce': loss_ce}

    def loss_frames(self, outputs, targets, indices, num_targets):
        """
        Regression loss for frame indices (L1)
        targets dicts must contain a "frames" key.
        Computes loss only for matched queries (ignores background).
        """
        src_frames = outputs['frames']  # (batch_size, num_queries)

        # Initialize lists to collect matched frames
        matched_src_frames = []
        matched_tgt_frames = []

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            # Check if the matched target is not background
            tgt_labels = targets[batch_idx]['labels'][tgt_idx]  # (num_matched_queries,)
            non_bg_mask = tgt_labels != 0  # (num_matched_queries,)

            # Filter out background matches
            if non_bg_mask.sum() == 0:
                continue  # No non-background matches in this batch element

            matched_src_frames.append(src_frames[batch_idx][src_idx][non_bg_mask])
            matched_tgt_frames.append(targets[batch_idx]['frames'][tgt_idx][non_bg_mask])

        if len(matched_src_frames) == 0:
            # No non-background matches in the entire batch
            loss_frame = torch.tensor(0.0, device=src_frames.device, requires_grad=True)
        else:
            # Concatenate all matched frames across the batch
            matched_src_frames = torch.cat(matched_src_frames)  # (total_non_bg_matched_queries,)
            matched_tgt_frames = torch.cat(matched_tgt_frames)  # (total_non_bg_matched_queries,)

            # Compute L1 loss only on matched queries
            loss_frame = F.l1_loss(matched_src_frames, matched_tgt_frames, reduction='mean')

        return {'loss_frame': loss_frame}

    def loss_xy(self, outputs, targets, indices, num_targets):
        """
        Regression loss for xy coordinates (L1)
        targets dicts must contain a "xy" key.
        Computes loss only for matched queries (ignores background).
        """
        src_xy = outputs['xy']  # (batch_size, num_queries, 2)

        # Initialize lists to collect matched xy coordinates
        matched_src_xy = []
        matched_tgt_xy = []

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            # Check if the matched target is not background
            tgt_labels = targets[batch_idx]['labels'][tgt_idx]  # (num_matched_queries,)
            non_bg_mask = tgt_labels != 0  # (num_matched_queries,)

            # Filter out background matches
            if non_bg_mask.sum() == 0:
                continue  # No non-background matches in this batch element

            matched_src_xy.append(src_xy[batch_idx][src_idx][non_bg_mask])  # (num_non_bg_matches, 2)
            matched_tgt_xy.append(targets[batch_idx]['xy'][tgt_idx][non_bg_mask])  # (num_non_bg_matches, 2)

        if len(matched_src_xy) == 0:
            # No non-background matches in the entire batch
            loss_xy = torch.tensor(0.0, device=src_xy.device, requires_grad=True)
        else:
            # Concatenate all matched xy coordinates across the batch
            matched_src_xy = torch.cat(matched_src_xy)  # (total_non_bg_matched_queries, 2)
            matched_tgt_xy = torch.cat(matched_tgt_xy)  # (total_non_bg_matched_queries, 2)

            # Compute L1 loss only on matched queries
            loss_xy = F.l1_loss(matched_src_xy, matched_tgt_xy, reduction='mean')

        return {'loss_xy': loss_xy}

    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Args:
            outputs (dict): Dict of tensors, must contain at least 'logits', 'frames', 'xy'.
            targets (list of dict): List of targets (len = batch_size), each target is a dict containing:
                "labels", "frames", "xy", "event_mask"
        Returns:
            dict: Losses
        """
        # Retrieve the matching between the outputs and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target events across all elements, for normalization purposes
        num_targets = sum([len(t['labels']) for t in targets])
        num_targets = max(num_targets, 1)

        # Compute all requested losses
        losses = {}
        for loss in self.losses:
            if loss == "labels":
                losses.update(self.loss_labels(outputs, targets, indices, num_targets))
            if loss == "frames":
                losses.update(self.loss_frames(outputs, targets, indices, num_targets))
            if loss == "xy":
                losses.update(self.loss_xy(outputs, targets, indices, num_targets))

        # Total loss weighted sum
        weighted_losses = {k: v * self.weight_dict[k] for k, v in losses.items() if k in self.weight_dict}
        total_loss = sum(weighted_losses.values())
        weighted_losses["loss"] = total_loss

        return weighted_losses
