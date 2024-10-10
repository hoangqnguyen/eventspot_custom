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
            cost_class = -prob[:, tgt_labels]  # (num_queries, num_targets)

            # Frame index cost: L1 distance between predicted and target frame indices
            cost_frame = torch.abs(frames.unsqueeze(1) - tgt_frames.unsqueeze(0))  # (num_queries, num_targets)

            # Coordinate cost: L1 distance between predicted and target xy coordinates
            cost_coord = torch.abs(xy.unsqueeze(1) - tgt_xy.unsqueeze(0)).sum(-1)  # (num_queries, num_targets)

            # Total cost
            cost_matrix = self.cost_class * cost_class + self.cost_frame * cost_frame + self.cost_coord * cost_coord  # (num_queries, num_targets)

            # Convert cost_matrix to numpy for scipy
            cost_matrix = cost_matrix.detach().cpu().numpy()

            # Perform Hungarian matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64),
                            torch.as_tensor(col_ind, dtype=torch.int64)))

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
        """Classification loss (CrossEntropy)
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
        """Regression loss for frame indices (L1)
        targets dicts must contain a "frames" key.
        """
        src_frames = outputs['frames']  # (batch_size, num_queries)

        # Flatten to (batch_size*num_queries,)
        src_frames = src_frames.view(-1)

        # Initialize target frames with zeros (background)
        target_frames = torch.zeros_like(src_frames)

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            # Assign target frame indices to matched queries
            target_frames[batch_idx * outputs['frames'].shape[1] + src_idx] = targets[batch_idx]['frames'][tgt_idx]

        # Compute L1 loss
        loss_frame = F.l1_loss(src_frames, target_frames, reduction='mean')

        return {'loss_frame': loss_frame}

    def loss_xy(self, outputs, targets, indices, num_targets):
        """Regression loss for xy coordinates (L1)
        targets dicts must contain a "xy" key.
        Applies mask to ignore background events.
        """
        src_xy = outputs['xy']  # (batch_size, num_queries, 2)

        # Flatten to (batch_size*num_queries, 2)
        src_xy = src_xy.view(-1, 2)

        # Initialize target xy with zeros (background)
        target_xy = torch.zeros_like(src_xy)

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            # Assign target xy coordinates to matched queries
            target_xy[batch_idx * outputs['xy'].shape[1] + src_idx] = targets[batch_idx]['xy'][tgt_idx]

        # Compute L1 loss
        loss_xy = F.l1_loss(src_xy, target_xy, reduction='mean')

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
        # Retrieve the matching between the outputs an
