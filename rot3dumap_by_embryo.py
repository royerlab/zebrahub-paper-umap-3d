"""
This script generates rotating 3D UMAP videos by higlighting each embryo
"""
import os
from os.path import join, exists
import string
from tqdm import tqdm
from joblib import delayed, Parallel
from natsort import natsorted
import numpy as np
import pandas as pd
import napari
import seaborn as sns
from napari_animation import Animation
import matplotlib.pyplot as plt
import colorcet as cc


# Colormap name
colormap_name = 'crest'
greys = np.array(sns.color_palette('Greys', 100)[0] + (0.15,)).reshape(1, -1)

# Define save path
savepath = join('output', 'by_fish_timepoint', colormap_name + '_flat')
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
if colormap_name == 'isolum':
    cmap = sns.color_palette(cc.isolum, 256)[::25][::-1]
else:
    cmap = sns.color_palette(colormap_name, len(uniq_time))
    cmap = [tuple((i ** 0.7)) for i in np.array(cmap)]

# Set parameters for the video
div = 3
scale_factor = 1  # scale factor for the final output video size
fps = 60  # frames per second for the final output video
nb_steps = fps * div  # number of steps between two target angles


"""
Generate a rotating UMAP with all timepoints with different colors
"""
print('Generating all timepoints...')
lab_color = np.zeros((len(df_umap), 4))
for i0, tp in enumerate(uniq_time):
    ind = df_meta_time.get_group(tp)
    lab_color[ind.index] = np.array(cmap[i0] + (1,)).reshape(1, -1)

viewer = napari.view_points(
    df_umap[['UMAP1', 'UMAP2', 'UMAP3']],
    scale=(100,) * 3,
    shading='none',
    size=0.06,
    name='umap3d',
    edge_width=0,
    face_color=lab_color,
    ndisplay=3,
)
viewer.window.resize(1000 + 300, 1000)

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

# generate legend
plt.style.use('dark_background')
fig, ax = plt.subplots(1)
legendFig = plt.figure(figsize=(1.8, 2.4))
plist = []
leg_names = []
alphabets, digits = string.ascii_lowercase, string.digits
for i, n in enumerate(uniq_time):
    num = n.rstrip(alphabets).rstrip()
    if num[0] == '0':
        num = num[1:]
    nam = n.strip(digits).strip()
    if num == '':
        n = nam
    else:
        n = ' '.join([num, nam])

    plist.append(
        ax.scatter(i, i, c=np.array(cmap[i] + (1,)).reshape(1, -1), s=40, label=n)
    )
    leg_names.append(n)
legendFig.legend(plist, leg_names, loc='center', frameon=False)
legendFig.savefig(join(savepath, 'legend_alltp.png'), dpi=300)
legendFig.savefig(join(savepath, 'legend_alltp_transparent.png'), dpi=300, transparent=True)


"""
Generate a rotating UMAP with global annotation
"""
print('Generating global annotation...')
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

# generate legend
plt.style.use('dark_background')
fig, ax = plt.subplots(1)
legendFig = plt.figure(figsize=(2.4, 2.4))
plist = []
leg_names = []
alphabets, digits = string.ascii_lowercase, string.digits
for i, n in enumerate(df_meta2_ga.groups.keys()):
    n = n.replace('_', ' ')
    plist.append(
        ax.scatter(i, i, c=np.array(cmap[i] + (1,)).reshape(1, -1), s=40, label=n)
    )
    leg_names.append(n)
legendFig.legend(plist, leg_names, loc='center', frameon=False)
legendFig.savefig(join(savepath, 'legend_global_annotation.png'), dpi=300)
legendFig.savefig(join(savepath, 'legend_global_annotation_transparent.png'), dpi=300, transparent=True)


"""
Generate 180 + 60 deg rotation for each time point with different colors
"""
print('Generating each timepoint...')


def single_proc(i0, tp):
    # Make a colormap
    ind = df_meta_time.get_group(tp)
    ind1 = df_meta.index.isin(ind.index)
    viewer = napari.view_points(
        df_umap[['UMAP1', 'UMAP2', 'UMAP3']][~ind1],
        scale=(100,) * 3,
        shading='none',
        size=0.06,
        name='others',
        edge_width=0,
        face_color=greys,
        ndisplay=3,
    )
    viewer.add_points(
        df_umap[['UMAP1', 'UMAP2', 'UMAP3']][ind1],
        scale=(100,) * 3,
        shading='none',
        size=0.06,
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


"""
Generate 180 + 60 deg rotation for each time point with only one embryo that has the largest data points
"""
print('Generating each timepoint with single embryo...')


def single_embryo(i0, tp):
    # Make a colormap
    ind = df_meta_time.get_group(tp)
    ind_counts = ind.value_counts('fish')
    fish_id = ind_counts.index[ind_counts.argmax()]
    ind1 = df_meta['fish'] == fish_id
    viewer = napari.view_points(
        df_umap[['UMAP1', 'UMAP2', 'UMAP3']][~ind1],
        scale=(100,) * 3,
        shading='none',
        size=0.06,
        name='others',
        edge_width=0,
        face_color=greys,
        ndisplay=3,
    )
    viewer.add_points(
        df_umap[['UMAP1', 'UMAP2', 'UMAP3']][ind1],
        scale=(100,) * 3,
        shading='none',
        size=0.06,
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


# generate legend
plt.style.use('dark_background')
fig, ax = plt.subplots(1)
legendFig = plt.figure(figsize=(2.4, 2.4))
plist = []
leg_names = []
alphabets, digits = string.ascii_lowercase, string.digits
for i, tp in enumerate(uniq_time):
    num = tp.rstrip(alphabets).rstrip()
    if num[0] == '0':
        num = num[1:]
    nam = tp.strip(digits).strip()
    if num == '':
        n = nam
    else:
        n = ' '.join([num, nam])

    ind = df_meta_time.get_group(tp)
    ind_counts = ind.value_counts('fish')
    fish_id = ind_counts.index[ind_counts.argmax()]
    n = f'{n} ({fish_id})'

    plist.append(
        ax.scatter(i, i, c=np.array(cmap[i] + (1,)).reshape(1, -1), s=40, label=n)
    )
    leg_names.append(n)
legendFig.legend(plist, leg_names, loc='center', frameon=False)
legendFig.savefig(join(savepath, 'legend_alltp_1embryo.png'), dpi=300)
legendFig.savefig(join(savepath, 'legend_alltp_1embryo_transparent.png'), dpi=300, transparent=True)
