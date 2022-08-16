from os.path import join
import numpy as np
import pandas as pd
import napari
import seaborn as sns
from napari_animation import Animation


# Colormap name
colormap_name = 'hls'

# Load data
data = pd.read_csv('seurat_combined_ALL_UMAP3D.csv')

# Make a color matrix for the annotations:
uniq_uniorg = data['timepoint'].unique()
lab_color = np.zeros((len(data), 4))
cmap = sns.color_palette(colormap_name, len(uniq_uniorg))
for i, fmly in enumerate(uniq_uniorg):
    c = np.array(cmap[i] + (1,)).reshape(1, -1)
    ind = data['timepoint'] == fmly
    lab_color[ind] = c

viewer = napari.view_points(
    data[['UMAP_1', 'UMAP_2', 'UMAP_3']],
    scale=(100,) * 3,
    shading='spherical',
    size=0.06,
    name='umap3d',
    edge_width=0,
    face_color=lab_color,
    ndisplay=3,
)

# Resizes window and tries to get a square canvas:
viewer.window.resize(1000+300, 1000)


# Set parameters for the video
scale_factor = 1  # scale factor for the final output video size
nb_steps = 60  # number of steps between two target angles

# Instantiates a napari animation object for our viewer:
animation = Animation(viewer)

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
animation.animate(f'3DUMAP_{colormap_name}.gif', canvas_only=True, fps=60, scale_factor=scale_factor)


# Make a legend
import matplotlib.pyplot as plt

plt.style.use('dark_background')
fig, ax = plt.subplots(1)
legendFig = plt.figure(figsize=(1.8, 2.4))
plist = []
i = 0
leg_names = []
for fmly in uniq_uniorg:
    plist.append(
        ax.scatter(i, i, c=np.array(cmap[i] + (1,)).reshape(1, -1), s=40, label=fmly)
    )
    i += 1
    leg_names.append(fmly)
legendFig.legend(plist, leg_names, loc='center', frameon=False)
legendFig.savefig(f'legend_{colormap_name}.png', dpi=300)

