# DSNeRF

Steps to use this code:

### Generate camera poses and sparse depth information using COLMAP

```
python imgs2poses.py <your_scenedir>
```

check http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIG19/lffusion/testscene.zip (NeRF-LLFF data) for directory structure.


### Training

```
python run_nerf.py --config configs/fern_dsnerf.txt
```


