# dataset.py

import os
import json
import random
from concurrent.futures import ThreadPoolExecutor
from transform import get_cpu_transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from tabulate import tabulate
from torchvision import tv_tensors
import numpy as np
from PIL import Image


def load_image_torchvision(frame_file, frames_dir):
    """
    Loads an image using torchvision's read_image function.

    Args:
        frame_file (str): Relative path to the image file (including subdirectory).
        frames_dir (str): Root directory containing all frames.

    Returns:
        torch.Tensor: Image tensor of shape (C, H, W).
    """
    frame_path = os.path.join(frames_dir, frame_file)
    if not os.path.exists(frame_path):
        raise FileNotFoundError(f"Image file not found: {frame_path}")
    image = read_image(frame_path)  # Read image as (C, H, W) tensor
    return image


class VolleyballVideoDataset(Dataset):
    def __init__(
        self,
        json_file,
        frames_dir,
        classes_file,
        transform=None,
        window_size=16,
        stride=8,
        num_events=10,  # Target number of events per sequence
        frame_size=(224, 224),
    ):
        """
        Initializes the VolleyballVideoDataset.

        Args:
            json_file (str): Path to the JSON file with annotations (train.json, val.json, test.json).
            frames_dir (str): Directory with all the frames.
            classes_file (str): Path to the classes file.
            transform (callable, optional): Optional transform to be applied on a sample.
            window_size (int): Number of frames per sequence.
            stride (int): Stride for the sliding window.
            num_events (int): Target number of events per sequence.
            frame_size (tuple): Size to resize frames.
        """
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.frame_size = frame_size
        self.frames_dir = frames_dir

        # Load class names, including 'background' as index 0
        with open(classes_file, "r") as f:
            class_names = f.read().splitlines()
        self.class_names = ["background"] + class_names
        self.num_classes = len(self.class_names)
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.class_names)
        }

        # Load video data from JSON
        with open(json_file, "r") as f:
            self.video_infos = json.load(f)

        self.sequences = self._build_sequences()

        # Determine the maximum number of events per sequence for padding
        events_per_sequence = [len(seq["seq_events"]) for seq in self.sequences]
        max_events_per_sequence = max(events_per_sequence) if events_per_sequence else 0
        self.num_events = max(num_events, max_events_per_sequence)

    def _build_sequences(self):
        """
        Builds sequences from the video data using a sliding window.

        Returns:
            list: List of sequence dictionaries.
        """
        sequences = []
        for video_info in self.video_infos:
            video_name = video_info["video"]
            num_frames = video_info["num_frames"]
            frames_path = os.path.join(self.frames_dir, video_name)
            if not os.path.isdir(frames_path):
                raise NotADirectoryError(f"Frames directory not found: {frames_path}")
            frame_files = sorted(os.listdir(frames_path))

            # Build event list
            events = []
            for event in video_info.get("events", []):
                frame_idx = event["frame"]
                label = event["label"]
                xy = event["xy"]
                events.append({"frame": frame_idx, "label": label, "xy": xy})

            # Create sequences using sliding window
            for start_idx in range(0, num_frames, self.stride):
                end_idx = min(start_idx + self.window_size, num_frames)
                seq_frame_files = frame_files[start_idx:end_idx]
                seq_length = len(seq_frame_files)

                # Adjust frame indices to be relative to the sequence
                seq_events = []
                for event in events:
                    if start_idx <= event["frame"] < end_idx:
                        event_in_seq = {
                            "frame": (event["frame"] - start_idx)
                            / seq_length,  # Normalize frame index
                            "label": event["label"],
                            "xy": event["xy"],
                        }
                        seq_events.append(event_in_seq)

                # Prepend video_name to frame files for correct path
                seq_frame_files_full = [
                    os.path.join(video_name, f) for f in seq_frame_files
                ]

                sequences.append(
                    {
                        "video_name": video_name,
                        "seq_frame_files": seq_frame_files_full,  # Full relative paths
                        "seq_events": seq_events,
                        "seq_length": seq_length,
                    }
                )

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieves the sequence at the specified index.

        Args:
            idx (int): Index of the sequence.

        Returns:
            dict: Dictionary containing images, frame indices, labels, and bounding boxes.
        """
        sequence_info = self.sequences[idx]
        seq_frame_files = sequence_info["seq_frame_files"]
        seq_length = sequence_info["seq_length"]

        # Load images in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            images = list(
                executor.map(
                    load_image_torchvision,
                    seq_frame_files,
                    [self.frames_dir] * len(seq_frame_files),
                )
            )

        images = torch.stack(images)  # Stack images into a tensor (seq_length, C, H, W)

        height, width = images.shape[-2:]

        # Prepare targets (bounding boxes and labels) for the entire sequence
        target = {
            "bounding_boxes": [],
            "labels": [],
            "frames": [],
        }

        # Collect events and their corresponding bounding boxes
        event_frames = set()  # To track which frames already have events
        for event in sequence_info["seq_events"]:
            frame_idx = event["frame"]  # Normalized frame index
            label_idx = self.class_to_idx.get(
                event["label"], 0
            )  # Convert label to index
            xy = event["xy"]  # (x, y) coordinates in relative terms (0 to 1)
            # Duplicate coordinates to create a bounding box (x1, y1, x2, y2)
            x1, y1 = xy[0] * width, xy[1] * height

            target["bounding_boxes"].append([x1, y1, x1, y1])
            target["labels"].append(label_idx)
            target["frames"].append(
                frame_idx
            )  # Keep the normalized frame index separately
            event_frames.add(frame_idx)  # Mark this frame as having an event

        # **Generate Background Events** if the number of events is less than num_events
        num_gt_events = len(target["labels"])
        num_background_events = max(0, self.num_events - num_gt_events)
        available_frame_indices = [
            i / seq_length
            for i in range(seq_length)
            if (i / seq_length) not in event_frames
        ]

        # Handle cases where available_frame_indices < num_background_events
        num_background_events = min(num_background_events, len(available_frame_indices))
        background_frame_indices = random.sample(
            available_frame_indices,
            num_background_events,
        )

        for frame_idx in background_frame_indices:
            target["bounding_boxes"].append([0.0, 0.0, 0.0, 0.0])  # Dummy box
            target["labels"].append(self.class_to_idx["background"])
            target["frames"].append(frame_idx)

        # Convert bounding boxes to TVTensors (in XYXY format)
        target["bounding_boxes"] = tv_tensors.BoundingBoxes(
            torch.tensor(target["bounding_boxes"], dtype=torch.float32),
            format="XYXY",
            canvas_size=(height, width),
        )
        target["labels"] = torch.tensor(target["labels"], dtype=torch.long)
        target["frames"] = torch.tensor(
            target["frames"], dtype=torch.float32
        )  # Normalized frame indices

        # Apply transformations to the images and target
        if self.transform:
            images, target = self.transform(images, target)

        # After the transformation, re-normalize the bounding boxes (xyxy format)
        new_height, new_width = images.shape[2], images.shape[3]
        bboxes_abs = target["bounding_boxes"].clone()

        # Normalize the bounding boxes
        bboxes_abs[..., [0, 2]] /= new_width  # Normalize x1 and x2
        bboxes_abs[..., [1, 3]] /= new_height  # Normalize y1 and y2

        # Update target with normalized bounding boxes
        target["bounding_boxes"] = bboxes_abs

        return {
            "images": images,  # Tensor of shape (seq_length, C, H, W)
            "frame": target["frames"],  # Normalized frame indices
            "label": target["labels"],  # Labels tensor
            "xy": bboxes_abs[:, :2],  # Bounding boxes (normalized XY format)
        }

    def print_stats(self):
        """
        Prints dataset statistics using tabulate for clarity.
        """
        total_sequences = len(self.sequences)
        total_frames = sum(seq["seq_length"] for seq in self.sequences)
        total_events = sum(len(seq["seq_events"]) for seq in self.sequences)
        events_per_sequence = [len(seq["seq_events"]) for seq in self.sequences]

        # Calculate min and max events per sequence
        min_events_per_sequence = min(events_per_sequence) if events_per_sequence else 0
        max_events_per_sequence = max(events_per_sequence) if events_per_sequence else 0

        table_data = [
            ["Total Sequences", f"{total_sequences:,}"],
            ["Total Frames", f"{total_frames:,}"],
            ["Total Events", f"{total_events:,}"],
            ["Avg Frames/Sequence", f"{total_frames / total_sequences:.2f}"],
            ["Avg Events/Sequence", f"{total_events / total_sequences:.2f}"],
            ["Min Events/Sequence", f"{min_events_per_sequence:,}"],
            ["Max Events/Sequence", f"{max_events_per_sequence:,}"],
        ]

        print(f"\n{'=' * 40}")
        print(
            tabulate(
                table_data, headers=["Metric", "Value"], tablefmt="rounded_outline"
            )
        )
        print(f"{'=' * 40}")


def get_dataloaders(config):
    """
    Returns a DataLoader for the given dataset.

    Args:
        dataset (Dataset): Instance of VolleyballVideoDataset or any dataset.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data (default is True for training).
        num_workers (int): Number of workers for parallel data loading (default is 4).

    Returns:
        DataLoader: The DataLoader instance.
    """

    # Define transforms for training and validation
    train_transforms = get_cpu_transforms(
        split="train",
        frame_size=tuple(config.frame_size),
    )
    val_transforms = get_cpu_transforms(
        split="val",
        frame_size=tuple(config.frame_size),
    )

    # Instantiate training and validation datasets
    train_dataset = VolleyballVideoDataset(
        json_file=config.train_json,
        frames_dir=config.frames_dir,
        classes_file=config.classes_file,
        transform=train_transforms,
        window_size=config.window_size,
        stride=config.stride,
        num_events=config.num_events,
        frame_size=tuple(config.frame_size),
    )

    val_dataset = VolleyballVideoDataset(
        json_file=config.val_json,
        frames_dir=config.frames_dir,
        classes_file=config.classes_file,
        transform=val_transforms,
        window_size=config.window_size,
        stride=config.stride,
        num_events=config.num_events,
        frame_size=tuple(config.frame_size),
    )

    # Print dataset statistics
    print("\nTraining Dataset Statistics:")
    train_dataset.print_stats()

    print("\nValidation Dataset Statistics:")
    val_dataset.print_stats()

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        # pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        # pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
        collate_fn=custom_collate_fn,
    )

    test_loader = None  # TODO

    return train_loader, val_loader, test_loader


def custom_collate_fn(batch):
    """
    Custom collate function to handle batches with variable sequence lengths and events.

    Args:
        batch (list): List of samples from the dataset.

    Returns:
        dict: Batched data.
    """
    images_list = [sample["images"] for sample in batch]
    frames_list = [sample["frame"] for sample in batch]  # Normalized frame indices
    labels_list = [sample["label"] for sample in batch]
    xys_list = [
        sample["xy"] for sample in batch
    ]  # Bounding boxes in normalized XY format

    # Find the maximum sequence length in the batch
    seq_lengths = [images.shape[0] for images in images_list]
    max_seq_length = max(seq_lengths)

    # Pad images to the maximum sequence length in the batch
    padded_images_list = []
    for images in images_list:
        seq_length = images.shape[0]
        if seq_length < max_seq_length:
            padding = torch.zeros((max_seq_length - seq_length, *images.shape[1:]))
            images = torch.cat([images, padding], dim=0)
        padded_images_list.append(images)
    batch_images = torch.stack(
        padded_images_list
    )  # Shape: (batch_size, max_seq_length, C, H, W)

    # Find the maximum number of events in the batch
    num_events_list = [frames.shape[0] for frames in frames_list]
    max_num_events = max(num_events_list)

    padded_frames = []
    padded_labels = []
    padded_xys = []
    event_masks = []

    for frames, labels, bboxes in zip(frames_list, labels_list, xys_list):
        num_events = frames.shape[0]
        if num_events < max_num_events:
            pad_length = max_num_events - num_events
            frames = torch.cat(
                [frames, torch.zeros(pad_length)], dim=0
            )  # Pad frame indices
            labels = torch.cat(
                [labels, torch.full((pad_length,), -1, dtype=torch.long)], dim=0
            )
            bboxes = torch.cat(
                [bboxes, torch.zeros((pad_length, bboxes.shape[1]))], dim=0
            )  # Pad bboxes
            mask = torch.cat([torch.ones(num_events), torch.zeros(pad_length)], dim=0)
        else:
            mask = torch.ones(num_events)

        padded_frames.append(frames)
        padded_labels.append(labels)
        padded_xys.append(bboxes)
        event_masks.append(mask)

    batch_frames = torch.stack(padded_frames)  # Shape: (batch_size, max_num_events)
    batch_labels = torch.stack(padded_labels)  # Shape: (batch_size, max_num_events)
    batch_xys = torch.stack(padded_xys)  # Shape: (batch_size, max_num_events, 2)
    batch_event_masks = torch.stack(event_masks)  # Shape: (batch_size, max_num_events)

    # Prepare the batch dictionary
    batch_dict = {
        "images": batch_images,  # Tensor of shape (batch_size, max_seq_length, C, H, W)
        "frame": batch_frames,  # Normalized frame indices (batch_size, max_num_events)
        "label": batch_labels,  # Event labels (batch_size, max_num_events)
        "xy": batch_xys,  # Bounding boxes (batch_size, max_num_events, 2)
        "event_mask": batch_event_masks,  # Mask for valid events (batch_size, max_num_events)
    }

    return batch_dict


def visualize_batch(batch, class_names):
    """
    Visualizes only the individual frames that contain non-background events from a batch of images.
    Event text labels are overlaid at the event locations using Matplotlib.

    Args:
        batch (dict): A batch dictionary returned by the DataLoader.
        class_names (list): A list of class names, where the index corresponds to the class label.
    """
    images = batch["images"]  # Shape: (batch_size, max_seq_length, C, H, W)
    frames = batch["frame"]  # Normalized frame indices (batch_size, max_num_events)
    labels = batch["label"]  # Event labels (batch_size, max_num_events)
    xys = batch["xy"]  # Event coordinates (batch_size, max_num_events, 2)
    event_mask = batch[
        "event_mask"
    ]  # Mask for valid events (batch_size, max_num_events)

    batch_size, max_seq_length, _, img_h, img_w = images.shape

    for i in range(batch_size):
        for e in range(len(labels[i])):
            if event_mask[i, e].item() == 1:
                label_idx = labels[i, e].item()
                if label_idx != 0:  # Skip background events
                    # Get the frame index and the corresponding image
                    frame_idx = int(frames[i, e].item() * (max_seq_length - 1))
                    img = images[i, frame_idx].permute(1, 2, 0).cpu().numpy()
                    img = (img - img.min()) / (
                        img.max() - img.min()
                    )  # Normalize to [0, 1]

                    # Get event coordinates and class label
                    x, y = xys[i, e].cpu().numpy()
                    x, y = x * img_w, y * img_h  # Scale normalized coordinates
                    label_text = class_names[label_idx]

                    # Plot the image with annotations
                    fig, ax = plt.subplots(1)
                    ax.imshow(img)
                    # Draw a circle at the event location
                    circ = patches.Circle(
                        (x, y), radius=5, linewidth=1, edgecolor="r", facecolor="none"
                    )
                    ax.add_patch(circ)
                    # Add label text
                    ax.text(
                        x,
                        y,
                        label_text,
                        fontsize=12,
                        color="yellow",
                        bbox=dict(facecolor="red", alpha=0.5),
                    )
                    plt.title(f"Batch {i+1}, Event {e+1}")
                    plt.axis("off")
                    plt.show()

    # Optionally, close all figures to free memory
    plt.close("all")


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2
    import utils

    configs = dict(
        json_file="data/kovo_288p/train.json",
        frames_dir="data/kovo_288p/frames",
        classes_file="data/kovo_288p/class.txt",
        transform=get_cpu_transforms(split="test"),
        window_size=64,
        stride=8,
        num_events=10,
        frame_size=(224, 224),
    )

    dataset = VolleyballVideoDataset(**configs)
    dataset.print_stats()
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=custom_collate_fn
    )

    with utils.Timer("Iterating over dataloader"):
        for batch in tqdm(dataloader):
            pass

    # batch = next(iter(dataloader))
    # for k, v in batch.items():
    #     print(k, v.shape)
    # visualize_batch(batch, dataset.class_names)
    # breakpoint()
    # for images, labels, coords, seq_mask in tqdm(dataloader):
    #     print(images.shape, labels.shape, coords.shape, seq_mask.shape)
