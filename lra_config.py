config = {
    "listops":{
        "pe_type": "ape", # "nope", "spe" or "ape"
        "vocab_size": 15 + 1 + 1, # 15 tokens + 1 PAD + 1 CLS
        "embed_size": 64,
        "max_seq_len": 1999 + 1,
        "num_head": 1,
        "proj_size": 32,
        "para_share_schema": "head", # "head", "kv", or "layer"
        "hidden_size": 64 * 4,
        "dataset_name": "listops",
        "pooling_type": "CLS", # "CLS", "MEAN", "SUM", or "FLATTEN"
        "encoder_dim": 64,
        "mlp_dim": 32,
        "num_class": 10,
        "interaction": "None",
        "enable_cuda": True,
        "device_id": 0,
        "embed_drop_prob": 0.0,
        "value_drop_prob": 0.0,
        "ffn_drop_prob": 0.0,
        "num_block": 3,
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.001,
        "epochs": 50,
        "optimizer": "adamw", # "adamw", "lion" or "tiger"
        "patience": 2,
        "num_workers": 2
    },
    "image":{
        "pe_type": "ape", # "nope", "spe" or "ape"
        "vocab_size": 256, # 256 unique pixel values
        "embed_size": 128,
        "max_seq_len": 1024,
        "num_head": 2,
        "proj_size": 32,
        "para_share_schema": "head", # "head", "kv", or "layer"
        "hidden_size": 128 * 4,
        "dataset_name": "image",
        "pooling_type": "FLATTEN", # "CLS", "MEAN", "SUM", or "FLATTEN"
        "encoder_dim": 128,
        "mlp_dim": 64,
        "num_class": 10,
        "interaction": "None", 
        "enable_cuda": True,
        "device_id": 0, # single GPU
        "embed_drop_prob": 0.0,
        "value_drop_prob": 0.0,
        "ffn_drop_prob": 0.0,
        "num_block": 3,
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.001,
        "epochs": 20,
        "optimizer": "adamw", # "adamw", "lion" or "tiger"
        "patience": 2,
        "num_workers": 2
    },
    "pathfinder":{
        "pe_type": "ape", # "nope", "spe" or "ape"
        "vocab_size": 225, # 225 unique pixel values
        "embed_size": 128,
        "max_seq_len": 1024,
        "num_head": 2,
        "proj_size": 32,
        "para_share_schema": "head", # "head", "kv", or "layer"
        "hidden_size": 128 * 4,
        "dataset_name": "pathfinder",
        "pooling_type": "FLATTEN", # "CLS", "MEAN", "SUM", or "FLATTEN"
        "encoder_dim": 128,
        "mlp_dim": 64,
        "num_class": 2,
        "interaction": "None",
        "enable_cuda": True,
        "device_id": 0,
        "embed_drop_prob": 0.0,
        "value_drop_prob": 0.0,
        "ffn_drop_prob": 0.0,
        "num_block": 3,
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.001,
        "epochs": 50,
        "optimizer": "adamw", # "adamw", "lion" or "tiger"
        "patience": 2,
        "num_workers": 2
    },
    "text":{
        "pe_type": "ape", # "nope", "spe" or "ape"
        "vocab_size": 95 + 1 + 1, # 95 unique symbols + 1 PAD + 1 CLS
        "embed_size": 128,
        "max_seq_len": 4096 + 1,
        "num_head": 2,
        "proj_size": 32,
        "para_share_schema": "head", # "head", "kv", or "layer"
        "hidden_size": 128 * 4,
        "dataset_name": "text",
        "pooling_type": "CLS", # "CLS", "MEAN", "SUM", or "FLATTEN"
        "encoder_dim": 128,
        "mlp_dim": 64,
        "num_class": 2,
        "interaction": "None",
        "enable_cuda": True,
        "device_id": 0,
        "embed_drop_prob": 0.1,
        "value_drop_prob": 0.0,
        "ffn_drop_prob": 0.1,
        "num_block": 3,
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.001,
        "epochs": 50,
        "optimizer": "adamw", # "adamw", "lion" or "tiger"
        "patience": 3,
        "num_workers": 2
    },
    "retrieval":{
        "pe_type": "ape", # "nope", "spe" or "ape"
        "vocab_size": 96 + 1 + 1, # 96 unique symbols + 1 PAD + 1 CLS
        "embed_size": 128,
        "max_seq_len": 4000 + 1,
        "num_head": 2,
        "proj_size": 32,
        "para_share_schema": "head", # "head", "kv", or "layer"
        "hidden_size": 128 * 4,
        "n_layers": 1,
        "protocol": 'chord',
        "dataset_name": "retrieval",
        "pooling_type": "CLS", # "CLS", "MEAN", "SUM", or "FLATTEN"
        "encoder_dim": 128,
        "mlp_dim": 64,
        "num_class": 2,
        "interaction": "CAT", # "NLI" or "CAT"
        "enable_cuda": True,
        "device_id": 0,
        "embed_drop_prob": 0.0,
        "value_drop_prob": 0.0,
        "ffn_drop_prob": 0.1,
        "num_block": 3,
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.001,
        "epochs": 80,
        "optimizer": "adamw", # "adamw", "lion" or "tiger"
        "patience": 3,
        "num_workers": 2
    }
}