# configs/base_kovo.py

from easydict import EasyDict as edict
import torch  # Ensure torch is imported for device

config = edict({
    # **Data Parameters**
    "train_json": "data/kovo_288p/train.json",
    "val_json": "data/kovo_288p/val.json",
    "frames_dir": "data/kovo_288p/frames",
    "classes_file": "data/kovo_288p/class.txt",

    # **Training Parameters**
    "batch_size": 8,
    "val_batch_size": 4,
    "num_workers": 8,
    "learning_rate": 3e-4,
    "max_epochs": 50,
    "seed": 42,
    "gradient_accumulation_steps": 8,

    # **Model Parameters**
    "num_queries": 12,
    "backbone_name": "regnety_002",
    "pretrained": True,
    "hidden_dim": 256,
    "nheads": 8,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "dim_feedforward": 512,
    "dropout": 0.1,

    # **Loss Parameters**
    "cost_class": 1.0,
    "cost_frame": 1.0,
    "cost_coord": 1.0,
    "eos_coef": 0.1,

    # **Dataset Parameters**
    "window_size": 64,
    "stride": 32,
    "num_events": 10,
    "frame_size": [224, 224],  # [Height, Width]

    # **Logging and Checkpointing**
    "log_dir": "logs/",
    "checkpoint_dir": "exp/checkpoints/",
    "save_top_k": 3,
    "early_stop_patience": 10,
    

    # **Others**
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # **Loss Weight Dictionary**
    "weight_dict": {
        "loss_ce": 1.0,
        "loss_frame": 1.0,
        "loss_xy": 1.0,
    },
})
