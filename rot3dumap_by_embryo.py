"""
This script generates rotating 3D UMAP videos by higlighting each embryo
"""
from os.path import join

from natsort import natsorted
import numpy as np
import pandas as pd
import napari
import seaborn as sns
from napari_animation import Animation


# Colormap name
colormap_name = 'hls'

# Load UMAP coordinates and their metadata
df_umap = pd.read_csv('umap_coords.csv')
df_meta = pd.read_csv('meta_data.csv')

# Sort fish name by time point
df_meta_time = df_meta.groupby('timepoint')
uniq_time = natsorted(df_meta.timepoint.unique())
uniq_time = uniq_time[:1] + [t for t in uniq_time if 'somite' in t] + [t for t in uniq_time if 'dpf' in t]

# Set parameters for the video
scale_factor = 1  # scale factor for the final output video size
nb_steps = 60  # number of steps between two target angles


# Generate scatter plot on napari
viewer = napari.Viewer()
viewer.window.resize(1000+300, 1000)

# Instantiates a napari animation object for our viewer:
animation = Animation(viewer)

for tp in uniq_time:
    ind = df_meta_time.get_group(tp)
    ind_gp = ind.groupby('fish')
    cmap = sns.color_palette('tab10', len(ind_gp))
    for i, (n, indf) in enumerate(ind_gp):
        data = df_umap.loc[indf.index]
        viewer.add_points(
            data[['UMAP1', 'UMAP2', 'UMAP3']],
            scale=(100,) * 3,
            shading='spherical',
            size=0.06,
            name=f'{n} {tp}',
            edge_width=0,
            face_color=np.array(cmap[i] + (1,)).reshape(1, -1),
        )

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
            join('output', 'by_fish_timepoint', f'rot3DUMAP_{tp}.gif'),
            canvas_only=True,
            fps=60,
            scale_factor=scale_factor
        )


# # Make a legend
# import matplotlib.pyplot as plt
#
# plt.style.use('dark_background')
# fig, ax = plt.subplots(1)
# legendFig = plt.figure(figsize=(1.8, 2.4))
# plist = []
# i = 0
# leg_names = []
# for fmly in uniq_uniorg:
#     plist.append(
#         ax.scatter(i, i, c=np.array(cmap[i] + (1,)).reshape(1, -1), s=40, label=fmly)
#     )
#     i += 1
#     leg_names.append(fmly)
# legendFig.legend(plist, leg_names, loc='center', frameon=False)
# legendFig.savefig(f'legend_{colormap_name}.png', dpi=300)

