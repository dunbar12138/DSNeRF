# Depth-supervised NeRF: Fewer Views and Faster Training for Free

[**Project**](https://www.cs.cmu.edu/~dsnerf/) | [**Paper**(coming soon)]() | [**Model & Data**(coming soon)]()

We propose DS-NeRF (Depth-supervised Neural Radi-ance Fields), a model for learning neural radiance fields that takes advantage of depth supervised by 3D point clouds. 

---

## Quick Start

### Dependencies

Install requirements:
```
pip install -r requirements.txt
```

You will also need [COLMAP](https://github.com/colmap/colmap) installed to compute poses if you want to run on your own real data.

### Data

Download data for the example scene: `fern`
```
bash download_example_data.sh
```

To play with other scenes presented in the paper, download the data [here]().

### Pre-trained Models

You can download th pre-trained models [here](). Place the downloaded directory in `./logs` in order to test it later. See the following directory structure for an example:
```
├── logs 
│   ├── fern_test
│   ├── flower_test  # downloaded logs
│   ├── trex_test    # downloaded logs
```

### How to Run?

#### Generate camera poses and sparse depth information using COLMAP (optional)

This step is necessary only when you want to run on your own real data.

First, place your scene directory somewhere. See the following directory structure for an example:
```
├── data
│   ├── fern
│   ├── ├── images
│   ├── ├── ├── image001.png
│   ├── ├── ├── image002.png
```

To generate the poses and sparse point cloud:
```
python imgs2poses.py <your_scenedir>
```

#### Training

To train a DS-NeRF on the example `fern` dataset:
```
python run_nerf.py --config configs/fern_dsnerf.txt
```

You can create your own experiment configuration to try other datasets.

#### Testing

Once you have the experiment directory (downloaded or trained on your own) in `./logs`, 

- to render the test views:
```
python run_nerf.py --config configs/fern_dsnerf.txt --render_only
```

- to only compute the evaluation metrics:
```
python run_nerf.py --config configs/fern_dsnerf.txt --eval
```


---

## Credits

This code borrows heavily from [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch).
