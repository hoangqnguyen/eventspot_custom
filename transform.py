import torch
from torchvision.transforms import v2 as T_v2


def get_transforms(split, frame_size=(224, 224)):
    """
    Returns the transforms for the given dataset split.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        frame_size (tuple): The size to which frames will be resized (default is 224x224).

    Returns:
        callable: The composed transform.
    """
    if split == "train":
        transform = T_v2.Compose(
            [
                T_v2.RandomResizedCrop(size=frame_size, scale=(0.6, 1.0)),
                T_v2.RandomHorizontalFlip(p=0.5),
                T_v2.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                T_v2.RandomRotation(degrees=15),
                T_v2.ToDtype(torch.float32, scale=True),
                T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        # For 'val' and 'test' splits
        transform = T_v2.Compose(
            [
                T_v2.Resize(frame_size),
                T_v2.CenterCrop(frame_size),
                T_v2.ToDtype(torch.float32, scale=True),
                T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transform