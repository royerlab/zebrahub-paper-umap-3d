"""
This script generates rotating 3D UMAP videos by higlighting each embryo
"""
import os
from os.path import join, exists

from natsort import natsorted
import numpy as np
import pandas as pd
import napari
import seaborn as sns
from napari_animation import Animation


# Colormap name
colormap_name = 'hls'
# Define save path
savepath = join('output', 'by_fish_timepoint')
if not exists(savepath):
    os.makedirs(savepath)

# Load UMAP coordinates and their metadata
loadpath = 'zebrahub/final_objects/v1'
df_umap = pd.read_csv(join(loadpath, 'umap_coords.csv'))
df_meta = pd.read_csv(join(loadpath, 'meta_data.csv'))

# Sort fish name by time point
df_meta_time = df_meta.groupby('timepoint')
uniq_time = natsorted(df_meta.timepoint.unique())
uniq_time = uniq_time[:1] + [t for t in uniq_time if 'somite' in t] + [t for t in uniq_time if 'dpf' in t]

# Set parameters for the video
scale_factor = 1  # scale factor for the final output video size
nb_steps = 60  # number of steps between two target angles


# Generate scatter plot on napari
viewer = napari.view_points(
    df_umap[['UMAP1', 'UMAP2', 'UMAP3']],
    scale=(100,) * 3,
    shading='spherical',
    size=0.06,
    name='umap3d',
    edge_width=0,
    face_color=np.zeros((1, 4)),
    ndisplay=3,
)
viewer.window.resize(1000+300, 1000)


for tp in uniq_time:
    print(f'Shooting the time point {tp}...')
    # Instantiates a napari animation object for our viewer:
    animation = Animation(viewer)

    # Make a colormap
    ind = df_meta_time.get_group(tp)
    ind_gp = ind.groupby('fish')
    cmap = sns.color_palette('tab10', len(ind_gp))
    lab_color = np.zeros((len(df_umap), 4))
    for i, (n, indf) in enumerate(ind_gp):
        lab_color[indf.index] = np.array(cmap[i] + (1,)).reshape(1, -1)
    viewer.layers[0].face_color = lab_color

    # Ensures we are in 3D view mode:
    viewer.dims.ndisplay = 3
    # resets the camera view:
    viewer.reset_view()

    # Start recording key frames after changing viewer state:
    viewer.camera.angles = (0.0, 0.0, 90.0)
    animation.capture_keyframe()
    viewer.camera.angles = (0.0, 180.0, 90.0)
    animation.capture_keyframe(steps=nb_steps // 2)
    viewer.camera.angles = (0.0, 360.0, 90.0)
    animation.capture_keyframe(steps=nb_steps // 2)

    # Render animation as a GIF:
    animation.animate(
        join(savepath, f'rot3DUMAP_{tp}.gif'),
        canvas_only=True,
        fps=60,
        scale_factor=scale_factor
    )

