# physical_ai_av

This repository contains a python developer kit and documentation (in the form of a [wiki](https://github.com/NVlabs/physical_ai_av/wiki) and interactive [notebooks](notebooks/)) for working with the [NVIDIA Physical AI Autonomous Vehicles Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles), one of the largest, most geographically diverse collections of multi-sensor data empowering AV researchers to build the next generation of Physical AI based end-to-end driving systems.

## Installation & Setup
```
pip install physical_ai_av
```
To use this package to access the data hosted on Hugging Face, you'll need to:
- [Create a Hugging Face account](https://huggingface.co/join) (if you don't have one already).
- Login and agree to the NVIDIA Autonomous Vehicle Dataset License Agreement visible at the top of the [PhysicalAI AV dataset card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).
- Create a [User Access Token](https://huggingface.co/docs/hub/en/security-tokens) (if you don't have one already) and choose a method for [authentication](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).


## Quickstart Tutorial

First, create the dataset object and choose a clip by indexing into the dataset
```python
from physical_ai_av.dataset import PhysicalAIAVDatasetInterface
import numpy as np


ds = PhysicalAIAVDatasetInterface(token=True)
clip_id = ds.clip_index.index[0] # First clip in the dataset
```


## Camera Data
If you want to cache the data locally then run the following command, otherwise skip it

```python
chunk_id = ds.get_clip_chunk(clip_id)

ds.download_chunk_features(
    int(chunk_id),
    features=ds.features.CAMERA.ALL
)

# instead of .ALL you can choose a specific camera on the vehicle:

# - .CAMERA_CROSS_LEFT_120FOV
# - .CAMERA_CROSS_RIGHT_120FOV
# - .CAMERA_FRONT_TELE_30FOV
# - .CAMERA_FRONT_WIDE_120FOV
# - .CAMERA_REAR_LEFT_70FOV
# - .CAMERA_REAR_RIGHT_70FOV
# - .CAMERA_REAR_TELE_30FOV
```

If you did not cache the data, you will need to pass `maybe_stream=True` to the `get_clip_feature` method below. Then you can access frame like so

```python
reader = ds.get_clip_feature(clip_id, "camera_front_wide_120fov")
frame_indices = np.array([0, 1, 2]) # get first 3 frames
frames = reader.decode_images_from_frame_indices(frame_indices) # (N, H, W, C) numpy array
```

## Lidar Data
```python
chunk_id = ds.get_clip_chunk(clip_id)

ds.download_chunk_features(
    int(chunk_id),
    features=ds.features.LIDAR.LIDAR_TOP_360FOV
)

reader = ds.get_clip_feature(clip_id, "lidar_top_360fov") # dict
```
## Radar Data

```python
chunk_id = ds.get_clip_chunk(clip_id)

ds.download_chunk_features(
    int(chunk_id),
    features=ds.features.RADAR.ALL
)

reader = ds.get_clip_feature(clip_id, "radar_corner_front_left_srr_0") # dict

# instead of .ALL you can choose a specific radar on the vehicle. See the huggingface repo.

```

## Calibration Data

```python
chunk_id = ds.get_clip_chunk(clip_id)

ds.download_chunk_features(
    int(chunk_id),
    features=ds.features.CALIBRATION.ALL
)

reader = ds.get_clip_feature(clip_id, "sensor_extrinsics") # pandas dataframe

# Instead of .ALL you can choose specific data:

# - .CAMERA_INTRINSICS
# - .SENSOR_EXTRINSICS
# - .VEHICLE_DIMENSIONS
```

The second argument can be one of `sensor_extrinsics`, `camera_extrinsics`, or `vehicle_dimensions`. You can also use any one of these as a replacement to the `ALL` keyword when downloading to cache.

## EgoMotion Data

```python
chunk_id = ds.get_clip_chunk(clip_id)

ds.download_chunk_features(
    int(chunk_id),
    features=ds.features.LABELS.EGOMOTION
)

reader = ds.get_clip_feature(clip_id, "egomotion")

print(reader(0.27)) # get egomotion data at timestamp 0.27
```


# TorchCodec Support
Video decoding is an intensive process and is often the bottleneck in deep learning pipelines. TorchCodec offeres gpu accelerated decoders. To use hardware accelerated decoding, please follow the instructions to [install torchcodec](https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec). Once torchcodec is installed, you can use it like so:

```python
reader = ds.get_clip_feature(clip_id, "camera_front_wide_120fov", use_torch_codec=True)
```

Note that if a gpu is not found, we fall back to cpu.
