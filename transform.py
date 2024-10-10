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
                # **Geometric Transformations**
                T_v2.RandomChoice(
                    [
                        T_v2.RandomHorizontalFlip(p=0.5),
                        # T_v2.RandomVerticalFlip(p=0.3),  # Adding vertical flips
                        T_v2.RandomAffine(
                            degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10
                        ),
                        T_v2.RandomPerspective(distortion_scale=0.2, p=0.5),
                        T_v2.RandomZoomOut(fill=0, side_range=(1.0, 2.0), p=0.5),
                        nn.Identity(),
                    ]
                ),
                T_v2.Resize(frame_size),
                T_v2.RandomResizedCrop(
                    size=frame_size, scale=(0.6, 1.0), ratio=(0.75, 1.333)
                ),
                # **Color Transformations**
                T_v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                T_v2.RandomGrayscale(p=0.2),
                T_v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
                T_v2.RandomSolarize(threshold=128, p=0.2),
                # **Normalization**
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
