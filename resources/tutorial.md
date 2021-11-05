## How to use Depth-supervised Loss in your own project

Adding depth supervision loss to your model is easy. There are four steps: (1) Run COLMAP, (2) Modify the dataloader, (3) Generate the rays and then render the depths, (4) Calculate the depth-supervised loss.

#### Run COLMAP on the training views to get the sparse reconstruction

First, place your scene directory somewhere. See the following directory structure for an example:

```
├── data
│  ├── fern
│  ├── ├── images
│  ├── ├── ├── image001.png
│  ├── ├── ├── image002.png
```

To generate the poses and sparse point cloud:

```
python imgs2poses.py <your_scenedir>
```

#### Adapt the dataloader to your own use

We provide a dataloader [`load_colmap_depth(scenedir)`](https://github.com/dunbar12138/DSNeRF/blob/main/load_llff.py#L339) in `load_llff.py` to load the depth information as well as some other information we need to train a DS-NeRF. 

The dataloader  `load_colmap_depth(scenedir)` returns a list. Each element in this list is a dict corresponding to one camera view from the training set. 

For example, if we load a 2-view dataset, the function will return a list of length 2. Each element is a dict with keys: {"depth", "coord", "weight"}.

#### Generate the rays for key-points

We provide a function [`get_rays_by_coord_np(H, W, focal, c2w, coords)`](https://github.com/dunbar12138/DSNeRF/blob/5ed4c688bfad1fb52d7a9bf2b0a080762a75b608/run_nerf_helpers.py#L268) to generate the rays based on their 2D locations in the image. We can get the key-points coordinates `coords` from the dataloader we mentioned above. The other information required by this function should be common in any NeRF-based projects.

The function returns the origins and normalized directions of the rays in the world coordinate system.

#### Render the depths of rays

Implementations of NeRF rendering vary in different codebases. However, the overall idea is the same. Suppose we have N sampled points along the ray t_1, t_2, ..., t_n, the rendered color is given by:

![](http://latex.codecogs.com/gif.latex?\\hat{C}=\sum_{i=1}^NT_i(1-\exp{(-\sigma(t_i)\delta_i)})\mathbf{c}(t_i),)

<!-- $$
\hat C = \sum_{i=1}^N T_i (1-\exp{(-\sigma(t_i)\delta_i)})\mathbf{c}(t_i),
$$ -->
where 

![](http://latex.codecogs.com/gif.latex?T_i=\exp{(-\sum_{j=1}^{i-1}\sigma(t_j)\delta_i)},\quad\delta_i=t_{i+1}-t_i.)


Similarly, the rendered depth is given by:

![](http://latex.codecogs.com/gif.latex?\\hat{D}=\sum_{i=1}^NT_i(1-\exp{(-\sigma(t_i)\delta_i)})t_i.)
<!-- $$
\hat D = \sum_{i=1}^N T_i (1-\exp{(-\sigma(t_i)\delta_i)})t_i.
$$ -->
We implement these rendering equations in the function [`raw2outputs`](https://github.com/dunbar12138/DSNeRF/blob/5ed4c688bfad1fb52d7a9bf2b0a080762a75b608/run_nerf_helpers.py#L341) in `run_nerf_helpers.py`.

#### Calculate the depth-supervised loss

Once we have rendered the depths `rendered_depth` for those keypoints, we are now able to calculate the depth-supervised loss with the reference depths `target_depth`:

```
depth_loss = torch.mean(((rendered_depth - target_depth) ** 2) * ray_weights)
```

where `ray_weights` are obtained from the dataloader.



