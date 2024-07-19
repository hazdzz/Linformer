# Linformer

## About
The PyTorch implementation of Linformer from the paper [*Linformer: Self-Attention with Linear Complexity*](<https://arxiv.org/abs/2006.04768>).

## Citation
```
@article{wang2020linformer,
  title     = {Linformer: Self-Attention with Linear Complexity},
  author    = {Sinong Wang and Belinda Z. Li and Madian Khabsa and Han Fang and Hao Ma},
  year      = {2020},
  journal   = {arXiv preprint arXiv: Arxiv-2006.04768}
}
```

## Datasets
1. LRA: https://mega.nz/file/sdcU3RKR#Skl5HomJJldPBqI7vfLlSAX8VA0XKWiQSPX1E09dwbk

## Training Steps
1. Create a data folder:
```console
mkdir data
```

2. Download the dataset compressed archive
```console
wget $URL
```

3. Decompress the dataset compressed archive and put the contents into the data folder
```console
unzip $dataset.zip
mv $datast ./data/$datast
```

4. Run the main file
```console
python $dataset_main.py --task="$task"
```

## Requirements
To install requirements:
```console
pip3 install -r requirements.txt