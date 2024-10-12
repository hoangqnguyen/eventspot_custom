# model.py

import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
import timm
from transform import get_gpu_transforms
from loss import HungarianMatcher, SetCriterion
from tabulate import tabulate


class SimpleVideoTFModel(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        num_queries=5,
        backbone_name="regnety_002",
        pretrained=True,
        learning_rate=3e-4,
        learning_rate_backbone=3e-4,
        weight_decay=1e-4,
        lr_drop=200,
        backbone_out_channels=None,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        **config,
    ):
        """
        Initializes the SimpleVideoTFModel.

        Args:
            num_classes (int): Number of target classes for classification.
            num_queries (int): Number of event queries.
            backbone_name (str): Name of the backbone model from timm.
            pretrained (bool): Whether to use pretrained weights for the backbone.
            learning_rate (float): Learning rate for the optimizer.
            backbone_out_channels (int, optional): Number of output channels from the backbone.
                If None, it will be inferred from the backbone.
            hidden_dim (int): Dimension of the features after projection.
            nheads (int): Number of attention heads in the Transformer.
            num_encoder_layers (int): Number of layers in the Transformer encoder.
            num_decoder_layers (int): Number of layers in the Transformer decoder.
            dim_feedforward (int): Dimension of the feedforward network in Transformer layers.
            dropout (float): Dropout rate in Transformer layers.
            matcher (HungarianMatcher, optional): Matcher for assigning predictions to targets.
            weight_dict (dict, optional): Weights for different loss components.
            eos_coef (float): Weight for the no-object class in classification.
        """
        super(SimpleVideoTFModel, self).__init__()
        self.learning_rate = learning_rate
        self.learning_rate_backbone = learning_rate_backbone
        self.weight_decay = weight_decay
        self.lr_drop = lr_drop

        self.num_classes = num_classes
        self.num_queries = num_queries  # Number of events to predict per sequence
        self.hidden_dim = hidden_dim

        # Backbone CNN
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True, out_indices=[-1]
        )

        # Infer backbone output channels if not provided
        if backbone_out_channels is None:
            backbone_out_channels = self.backbone.feature_info.channels()[-1]

        # Projection layer to reduce feature dimension
        self.feature_proj = nn.Linear(backbone_out_channels, hidden_dim)

        # Transformer Encoder to process the sequence of features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False
        )

        # Learnable queries for the number of events
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

        # Transformer Decoder to map queries to event features
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.frame_embed = nn.Linear(hidden_dim, 1)
        self.coord_embed = nn.Linear(hidden_dim, 2)

        # Initialize Matcher and Criterion
        self.criterion = get_criterion(num_classes=num_classes, **config)

        self.gpu_transform = {
            "train": get_gpu_transforms("train"),
            "val": get_gpu_transforms("val"),
            "test": get_gpu_transforms("test"),
        }

    def print_model_stats(self):
        """
        Prints model statistics including number of parameters and relevant model configuration details.
        """
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_name = type(self.backbone).__name__
        stats = [
            ["Backbone Model", backbone_name],
            ["Number of Parameters", f"{num_params:,}"],
            ["Hidden Dimension", self.hidden_dim],
            ["Transformer Encoder Layers", self.transformer_encoder.num_layers],
            ["Transformer Decoder Layers", self.transformer_decoder.num_layers],
            ["Number of Queries (Events)", self.num_queries],
            ["Number of Classes", self.num_classes],
            ["Learning Rate", self.learning_rate],
        ]
        print(f"\n{'=' * 40}")
        print(f"Model Summary for {self.__class__.__name__}:")
        print(tabulate(stats, headers=["Attribute", "Value"], tablefmt="fancy_grid"))
        print(f"{'=' * 40}\n")

    def _apply_gpu_transform(self, x, mode="train"):
        xs = []
        for b in range(x.shape[0]):
            xs.append(self.gpu_transform[mode](x[b]))
        return torch.stack(xs, dim=0)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W).

        Returns:
            logits (torch.Tensor): Classification logits of shape (B, num_queries, num_classes).
            frames (torch.Tensor): Predicted frame indices of shape (B, num_queries).
            coords (torch.Tensor): Predicted xy coordinates of shape (B, num_queries, 2).
        """
        B, T, C, H, W = x.shape

        x = self._apply_gpu_transform(x, mode="train" if self.training else "val")

        # Reshape to (B*T, C, H, W) to process each frame individually
        x = x.view(B * T, C, H, W)  # (B*T, C, H, W)

        # Extract features using the backbone
        features = self.backbone(x)[
            0
        ]  # Assuming the last feature map, shape: (B*T, F, H', W')

        # Global Average Pooling to get a feature vector per frame
        features = (
            F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
        )  # (B*T, F)

        # Project features to a fixed dimension
        features = self.feature_proj(features)  # (B*T, hidden_dim)

        # Reshape back to (B, T, hidden_dim)
        features = features.view(B, T, self.hidden_dim)

        # Permute to (T, B, hidden_dim) for Transformer Encoder
        features = features.permute(1, 0, 2)  # (T, B, hidden_dim)

        # Pass through Transformer Encoder
        memory = self.transformer_encoder(features)  # (T, B, hidden_dim)

        # Prepare queries
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            1, B, 1
        )  # (num_queries, B, hidden_dim)

        # Transformer Decoder
        hs = self.transformer_decoder(
            query_embed, memory
        )  # (num_queries, B, hidden_dim)

        # Transpose to (B, num_queries, hidden_dim)
        hs = hs.permute(1, 0, 2)  # (B, num_queries, hidden_dim)

        # Prediction heads
        logits = self.class_embed(hs)  # (B, num_queries, num_classes)
        frames = self.frame_embed(hs).squeeze(-1)  # (B, num_queries)
        coords = self.coord_embed(hs)  # (B, num_queries, 2)

        return logits, frames, coords

    def common_step(self, batch, batch_idx):
        """
        Predicts the class labels, frame indices, and xy coordinates for the input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W).

        Returns:
            dict: Dictionary containing losses.
        """        
        images = batch["images"]  # (B, T, C, H, W)
        labels = batch["label"]  # (B, num_events)
        frames_gt = batch["frame"]  # (B, num_events)
        xy_gt = batch["xy"]  # (B, num_events, 2)
        event_mask = batch["event_mask"]  # (B, num_events)

        # Prepare targets for SetCriterion
        targets = []
        for b in range(images.shape[0]):
            target = {}
            valid = event_mask[b].bool()
            target["labels"] = labels[b][valid]  # (num_valid_events,)
            target["frames"] = frames_gt[b][valid]  # (num_valid_events,)
            target["xy"] = xy_gt[b][valid]  # (num_valid_events, 2)
            targets.append(target)

        # Forward pass
        logits, frames_pred, coords = self.forward(images)

        # Prepare outputs for SetCriterion
        outputs = {
            "logits": logits,  # (B, num_queries, num_classes)
            "frames": frames_pred.sigmoid(),  # (B, num_queries)
            "xy": coords.sigmoid(),  # (B, num_queries, 2)
        }

        # Compute loss
        loss_dict = self.criterion(outputs, targets)
        return loss_dict

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch (dict): Batch dictionary containing 'images', 'frame', 'label', 'xy', 'event_mask'.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        loss_dict = self.common_step(batch, batch_idx)

        # Log the losses
        for k, v in loss_dict.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch (dict): Batch dictionary containing 'images', 'frame', 'label', 'xy', 'event_mask'.
            batch_idx (int): Batch index.
        """
        loss_dict = self.common_step(batch, batch_idx)

        # Log the losses
        for k, v in loss_dict.items():
            self.log(k, v, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        """
        Configures the optimizer with separate learning rates for backbone and other layers.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        # Parameters with separate learning rate for the backbone
        backbone_params = list(self.backbone.parameters())
        non_backbone_params = [
            p for n, p in self.named_parameters() if not n.startswith("backbone")
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.learning_rate_backbone},
                {"params": non_backbone_params, "lr": self.learning_rate},
            ],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer, self.lr_drop
            ),
        }


def get_criterion(
    cost_class, cost_frame, cost_coord, eos_coef, num_classes, weight_dict, **kwargs
):
    """
    Returns the criterion for the given configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        SetCriterion: Criterion for the model.
    """
    # Initialize Matcher and Criterion
    matcher = HungarianMatcher(
        cost_class=cost_class,
        cost_frame=cost_frame,
        cost_coord=cost_coord,
    )

    criterion = SetCriterion(
        matcher=matcher,
        eos_coef=eos_coef,
        num_classes=num_classes,
        weight_dict=weight_dict,
        losses=["labels", "frames", "xy"],
    )

    return criterion


if __name__ == "__main__":
    num_classes = 10
    num_queries = 12  # Number of events to predict
    backbone_name = "regnety_002"
    pretrained = True
    hidden_dim = 256
    nheads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 512
    dropout = 0.1

    model = SimpleVideoTFModel(
        num_classes=num_classes,
        num_queries=num_queries,
        backbone_name=backbone_name,
        pretrained=pretrained,
        hidden_dim=hidden_dim,
        nheads=nheads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )

    model.print_model_stats()

    # Create dummy data
    batch_size = 2
    seq_length = 8
    num_events = 5  # For testing purposes

    # Assuming the backbone expects 224x224 images
    dummy_images = torch.randn(batch_size, seq_length, 3, 224, 224)  # (B, T, C, H, W)
    dummy_labels = torch.randint(
        1, num_classes, (batch_size, num_events)
    )  # (B, num_events), labels 1..(num_classes-1)
    dummy_frames = torch.randint(
        0, seq_length, (batch_size, num_events)
    ).float()  # (B, num_events)
    dummy_xy = torch.rand(batch_size, num_events, 2)  # (B, num_events, 2)
    dummy_event_mask = torch.ones(batch_size, num_events)  # (B, num_events)

    # Create dummy batch
    dummy_batch = {
        "images": dummy_images,
        "frame": dummy_frames,
        "label": dummy_labels,
        "xy": dummy_xy,
        "event_mask": dummy_event_mask,
    }

    # Forward pass
    logits, frames_pred, coords = model.common_step(dummy_batch["images"])
    print("Logits shape:", logits.shape)  # Expected: (2, num_queries, num_classes)
    print("Frames_pred shape:", frames_pred.shape)  # Expected: (2, num_queries)
    print("Coords shape:", coords.shape)  # Expected: (2, num_queries, 2)

    # Prepare targets
    targets = []
    for b in range(batch_size):
        target = {}
        valid = dummy_event_mask[b].bool()
        target["labels"] = dummy_labels[b][valid]  # (num_valid_events,)
        target["frames"] = dummy_frames[b][valid]  # (num_valid_events,)
        target["xy"] = dummy_xy[b][valid]  # (num_valid_events, 2)
        targets.append(target)

    # Prepare outputs
    outputs = {"logits": logits, "frames": frames_pred, "xy": coords}

    # Initialize Matcher and Criterion
    matcher = HungarianMatcher(cost_class=1.0, cost_frame=1.0, cost_coord=1.0)
    weight_dict = {"loss_ce": 1.0, "loss_frame": 1.0, "loss_xy": 1.0}
    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,
        num_classes=num_classes,
        losses=["labels", "frames", "xy"],
    )

    # Compute losses
    loss_dict = criterion(outputs, targets)
    print(loss_dict)
