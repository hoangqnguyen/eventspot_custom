import os
import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class VolleyballVideoDataset(Dataset):
    def __init__(self, json_file, frames_dir, classes_file, transform=None, window_size=16, stride=8, negative_sampling_ratio=1.0):
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.negative_sampling_ratio = negative_sampling_ratio

        # Load class names
        with open(classes_file, 'r') as f:
            self.class_names = f.read().splitlines()
            
        self.class_names = ['background'] + self.class_names
        self.num_classes = len(self.class_names)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}


        # Load video data
        with open(json_file, 'r') as f:
            self.video_infos = json.load(f)

        self.frames_dir = frames_dir

        # Build sequences using sliding window
        self.sequences = []
        for video_info in self.video_infos:
            video_name = video_info['video']
            num_frames = video_info['num_frames']
            frames_path = os.path.join(self.frames_dir, video_name)
            frame_files = sorted(os.listdir(frames_path))

            # Build event list
            events = []
            for event in video_info['events']:
                frame_idx = event['frame']
                label = event['label']
                xy = event['xy']
                events.append({
                    'frame': frame_idx,
                    'label': label,
                    'xy': xy
                })

            # Create sequences using sliding window
            for start_idx in range(0, num_frames, self.stride):
                end_idx = min(start_idx + self.window_size, num_frames)
                seq_frame_files = frame_files[start_idx:end_idx]
                seq_length = len(seq_frame_files)

                # Get events within this sequence
                seq_events = []
                for event in events:
                    if start_idx <= event['frame'] < end_idx:
                        # Adjust frame index to be relative to the sequence
                        event_in_seq = {
                            'frame': event['frame'] - start_idx,  # Relative frame index
                            'label': event['label'],
                            'xy': event['xy']
                        }
                        seq_events.append(event_in_seq)

                self.sequences.append({
                    'video_name': video_name,
                    'seq_frame_files': seq_frame_files,
                    'seq_events': seq_events,
                    'seq_length': seq_length,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'num_frames': num_frames
                })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_info = self.sequences[idx]
        seq_frame_files = sequence_info['seq_frame_files']
        seq_length = sequence_info['seq_length']

        # Load frames
        images = []
        for frame_file in seq_frame_files:
            frame_path = os.path.join(self.frames_dir, sequence_info['video_name'], frame_file)
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        images = torch.stack(images)  # Shape: (seq_length, C, H, W)
        
        # Prepare events data (positive samples)
        events = sequence_info['seq_events']

        frames = []
        labels = []
        xys = []

        for event in events:
            frame_idx = event['frame']
            frame_normalized = frame_idx / (seq_length - 1) if seq_length > 1 else 0.0
            label_idx = self.class_to_idx[event['label']]
            xy = event['xy']
            frames.append(frame_normalized)
            labels.append(label_idx)
            xys.append(xy)

        # Negative Sampling
        total_frames = sequence_info['seq_length']
        event_frames = set(event['frame'] for event in events)
        non_event_frames = set(range(total_frames)) - event_frames
        num_negative_samples = int(len(event_frames) * self.negative_sampling_ratio)

        if num_negative_samples > 0 and len(non_event_frames) > 0:
            sampled_negative_frames = random.sample(list(non_event_frames), min(num_negative_samples, len(non_event_frames)))
            for frame_idx in sampled_negative_frames:
                frame_normalized = frame_idx / (seq_length - 1) if seq_length > 1 else 0.0
                frames.append(frame_normalized)
                labels.append(self.class_to_idx['background'])
                xys.append([0.0, 0.0])  # Or any default value
        elif len(event_frames) == 0:
            # For sequences with no events, sample negative frames
            num_negative_samples = max(1, int(self.negative_sampling_ratio))
            sampled_negative_frames = random.sample(range(total_frames), min(num_negative_samples, total_frames))
            for frame_idx in sampled_negative_frames:
                frame_normalized = frame_idx / (seq_length - 1) if seq_length > 1 else 0.0
                frames.append(frame_normalized)
                labels.append(self.class_to_idx['background'])
                xys.append([0.0, 0.0])

        # Convert lists to tensors
        frames = torch.tensor(frames, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        xys = torch.tensor(xys, dtype=torch.float32)

        # Return sample
        sample = {
            'images': images,
            'frame': frames,
            'label': labels,
            'xy': xys
        }
        return sample

def custom_collate_fn(batch):
    """
    batch: List of samples (dicts) from the Dataset.
    """
    images_list = [sample['images'] for sample in batch]
    frames_list = [sample['frame'] for sample in batch]
    labels_list = [sample['label'] for sample in batch]
    xys_list = [sample['xy'] for sample in batch]

    # Pad images to the maximum sequence length in the batch
    seq_lengths = [images.shape[0] for images in images_list]
    max_seq_length = max(seq_lengths)

    padded_images_list = []
    for images in images_list:
        seq_length = images.shape[0]
        if seq_length < max_seq_length:
            pad_length = max_seq_length - seq_length
            padding = torch.zeros((pad_length, *images.shape[1:]))
            images = torch.cat([images, padding], dim=0)
        padded_images_list.append(images)
    batch_images = torch.stack(padded_images_list)  # Shape: (batch_size, max_seq_length, C, H, W)

    # Pad events (frames, labels, xy) to the maximum number of events in the batch
    num_events_list = [labels.shape[0] for labels in labels_list]
    max_num_events = max(num_events_list)

    padded_frames = []
    padded_labels = []
    padded_xys = []
    event_masks = []

    for frames, labels, xys in zip(frames_list, labels_list, xys_list):
        num_events = labels.shape[0]
        # Pad frames
        if num_events < max_num_events:
            pad_length = max_num_events - num_events
            frames = torch.cat([frames, torch.zeros(pad_length)], dim=0)
            labels = torch.cat([labels, torch.full((pad_length,), -1, dtype=torch.long)], dim=0)  # Use -1 for padding
            xys = torch.cat([xys, torch.zeros(pad_length, xys.shape[1])], dim=0)
            mask = torch.cat([torch.ones(num_events), torch.zeros(pad_length)], dim=0)
        else:
            mask = torch.ones(num_events)
        padded_frames.append(frames)
        padded_labels.append(labels)
        padded_xys.append(xys)
        event_masks.append(mask)

    batch_frames = torch.stack(padded_frames)  # Shape: (batch_size, max_num_events)
    batch_labels = torch.stack(padded_labels)  # Shape: (batch_size, max_num_events)
    batch_xys = torch.stack(padded_xys)        # Shape: (batch_size, max_num_events, 2)
    batch_event_masks = torch.stack(event_masks)  # Shape: (batch_size, max_num_events)

    # Prepare the batch dictionary
    batch_dict = {
        'images': batch_images,           # Tensor of shape (batch_size, max_seq_length, C, H, W)
        'frame': batch_frames,            # Tensor of shape (batch_size, max_num_events)
        'label': batch_labels,            # Tensor of shape (batch_size, max_num_events)
        'xy': batch_xys,                  # Tensor of shape (batch_size, max_num_events, 2)
        'event_mask': batch_event_masks   # Tensor of shape (batch_size, max_num_events)
    }

    return batch_dict


if __name__ == "__main__":
    from tqdm import tqdm

    configs = dict(
        json_file="data/kovo/train.json",
        frames_dir="data/kovo/frames",
        classes_file="data/kovo/class.txt",
        window_size=16,
        stride=8,
        transform=transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        ),
    )

    dataset = VolleyballVideoDataset(**configs)
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=custom_collate_fn
    )

    batch = next(iter(dataloader))
    breakpoint()
    # for images, labels, coords, seq_mask in tqdm(dataloader):
    #     print(images.shape, labels.shape, coords.shape, seq_mask.shape)
