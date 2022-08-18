"""
This script generates rotating 3D UMAP videos by higlighting each embryo
"""
import os
from os.path import join, exists

from tqdm import tqdm
from joblib import delayed, Parallel
from natsort import natsorted
import numpy as np
import pandas as pd
import napari
import seaborn as sns
from napari_animation import Animation


# Colormap name
colormap_name = 'crest'
greys = np.array(sns.color_palette('Greys', 100)[0] + (0.15,)).reshape(1, -1)

# Define save path
savepath = join('output', 'by_fish_timepoint', colormap_name)
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
cmap = sns.color_palette(colormap_name, len(uniq_time))

# Set parameters for the video
div = 3
scale_factor = 1  # scale factor for the final output video size
fps = 60  # frames per second for the final output video
nb_steps = fps * div  # number of steps between two target angles


# Generate a rotating UMAP with all timepoints with different colors
lab_color = np.zeros((len(df_umap), 4))
for i0, tp in enumerate(uniq_time):
    ind = df_meta_time.get_group(tp)
    lab_color[ind.index] = np.array(cmap[i0] + (1,)).reshape(1, -1)

viewer = napari.view_points(
    df_umap[['UMAP1', 'UMAP2', 'UMAP3']],
    scale=(100,) * 3,
    shading='spherical',
    size=0.06,
    name='umap3d',
    edge_width=0,
    face_color=lab_color,
    ndisplay=3,
)

# Instantiates a napari animation object for our viewer:
animation = Animation(viewer)

# Ensures we are in 3D view mode:
viewer.dims.ndisplay = 3
# resets the camera view:
viewer.reset_view()

# Start recording key frames after changing viewer state:
viewer.camera.angles = (0.0, 0.0, 90.0)
animation.capture_keyframe(steps=nb_steps)
viewer.camera.angles = (0.0, 180.0, 90.0)
animation.capture_keyframe(steps=nb_steps)
viewer.camera.angles = (0.0, 360.0, 90.0)
animation.capture_keyframe(steps=nb_steps)
viewer.camera.angles = (0.0, 60.0, 90.0)
animation.capture_keyframe(steps=nb_steps // 3)

# Render animation as a GIF:
animation.animate(
    join(savepath, f'rot3DUMAP_alltp.mov'),
    canvas_only=True,
    fps=fps,
    scale_factor=scale_factor
)


# Generate a rotating UMAP with global annotation
df_meta2 = pd.read_csv('zebrahub/final_objects/v2/meta_data.csv')
df_meta2_ga = df_meta2.groupby('global_annotation')
cmap2 = sns.color_palette('hls', len(df_meta2_ga))
lab_color = np.zeros((len(df_umap), 4))
for i, (n, ind) in enumerate(df_meta2_ga):
    lab_color[ind.index] = np.array(cmap2[i] + (1,)).reshape(1, -1)
viewer.layers[0].face_color = lab_color

# Instantiates a napari animation object for our viewer:
animation = Animation(viewer)

# Ensures we are in 3D view mode:
viewer.dims.ndisplay = 3
# resets the camera view:
viewer.reset_view()

# Start recording key frames after changing viewer state:
viewer.camera.angles = (0.0, 0.0, 90.0)
animation.capture_keyframe(steps=nb_steps)
viewer.camera.angles = (0.0, 180.0, 90.0)
animation.capture_keyframe(steps=nb_steps)
viewer.camera.angles = (0.0, 360.0, 90.0)
animation.capture_keyframe(steps=nb_steps)
viewer.camera.angles = (0.0, 60.0, 90.0)
animation.capture_keyframe(steps=nb_steps // 3)

# Render animation as a GIF:
animation.animate(
    join(savepath, f'rot3DUMAP_global.mov'),
    canvas_only=True,
    fps=fps,
    scale_factor=scale_factor
)


# Generate 180 + 60 deg rotation for each time point with different colors
def single_proc(i0, tp):
    # Make a colormap
    ind = df_meta_time.get_group(tp)
    ind1 = df_meta.index.isin(ind.index)
    viewer = napari.view_points(
        df_umap[['UMAP1', 'UMAP2', 'UMAP3']][~ind1],
        scale=(100,) * 3,
        shading='spherical',
        size=0.06,
        name='others',
        edge_width=0,
        face_color=greys,
        ndisplay=3,
    )
    viewer.add_points(
        df_umap[['UMAP1', 'UMAP2', 'UMAP3']][ind1],
        scale=(100,) * 3,
        shading='spherical',
        size=0.09,
        name=tp,
        edge_width=0,
        face_color=np.array(cmap[i0] + (1,)).reshape(1, -1),
        blending='translucent_no_depth',
    )
    viewer.window.resize(1000 + 300, 1000)

    animation = Animation(viewer)

    # Ensures we are in 3D view mode:
    viewer.dims.ndisplay = 3
    # resets the camera view:
    viewer.reset_view()

    # Start recording key frames after changing viewer state:
    for i in range(div + 2):
        viewer.camera.angles = (0.0, i * 180 // div + (i0 * 180), 90.0)
        animation.capture_keyframe(steps=nb_steps // div)

    # Render animation as a GIF:
    animation.animate(
        join(savepath, f'rot3DUMAP_{tp}.mov'),
        canvas_only=True,
        fps=fps,
        scale_factor=scale_factor
    )


_ = Parallel(n_jobs=10)(
    delayed(single_proc)(i, tp) for i, tp in enumerate(tqdm(uniq_time))
)


# Generate 180 + 60 deg rotation for each time point with only one embryo that has the largest data points
def single_embryo(i0, tp):
    # Make a colormap
    ind = df_meta_time.get_group(tp)
    ind_counts = ind.value_counts('fish')
    fish_id = ind_counts.index[ind_counts.argmax()]
    ind1 = df_meta['fish'] == fish_id
    viewer = napari.view_points(
        df_umap[['UMAP1', 'UMAP2', 'UMAP3']][~ind1],
        scale=(100,) * 3,
        shading='spherical',
        size=0.06,
        name='others',
        edge_width=0,
        face_color=greys,
        ndisplay=3,
    )
    viewer.add_points(
        df_umap[['UMAP1', 'UMAP2', 'UMAP3']][ind1],
        scale=(100,) * 3,
        shading='spherical',
        size=0.09,
        name=tp,
        edge_width=0,
        face_color=np.array(cmap[i0] + (1,)).reshape(1, -1),
        blending='translucent_no_depth',
    )
    viewer.window.resize(1000 + 300, 1000)

    animation = Animation(viewer)

    # Ensures we are in 3D view mode:
    viewer.dims.ndisplay = 3
    # resets the camera view:
    viewer.reset_view()

    # Start recording key frames after changing viewer state:
    for i in range(div + 2):
        viewer.camera.angles = (0.0, i * 180 // div + (i0 * 180), 90.0)
        animation.capture_keyframe(steps=nb_steps // div)

    # Render animation as a GIF:
    animation.animate(
        join(savepath, f'rot3DUMAP_{fish_id}_{tp}.mov'),
        canvas_only=True,
        fps=fps,
        scale_factor=scale_factor
    )


_ = Parallel(n_jobs=10)(
    delayed(single_embryo)(i, tp) for i, tp in enumerate(tqdm(uniq_time))
)


