# register3d

An efficient binary value 3d volume registration algorithm based on Cupy.

## Installation
```bash
pip install register3d
```

## Usage
```python
import register3d
import cupy as cp

# full_image3d and image_part3d are binary value 3d array
# value 1.0 means inside the surface
# value 0.0 means outside the surface
full_image3d = cp.array([ ... ]) # Your large image
image_part3d = cp.array([ ... ]) # Your small image

# match_3d_data_rotate will try to rotate image_part3d to match in full_image3d
# registered_image_part is of the same size as full_image3d
# less the score, better the match
# rotation_matrix is a rotation matrix not far from cp.eye(3)
registered_image_part, score, rotation_matrix = register3d.match_3d_data_rotate(full_image3d, image_part3d)
```
