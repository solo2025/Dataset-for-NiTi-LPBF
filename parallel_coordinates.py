import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
plot the parallel map of clustered data
"""


# df = pd.read_csv("infered_new_dataset_0706_clustered.csv")
df = pd.read_csv("clustered_data/clusted_full_dataset_0726.csv")

labels = "xNi,P/W,v/mm/s,h/um,t/um,RA,D/um,Oxygen/ppm,Ev/J.mm-3,RelativeDensity/%-pred,UTS/MPa-pred,EL/%-pred,SIM-SMR/MPa-pred,Ms/K-pred,Hysteresis/K-pred,cluster_labels".split(",")

df = df[labels]


fig = go.Figure(data=
            go.Parcoords(
                line = dict(color = df["cluster_labels"],
                        # colorscale = 'Geyser',       
                        # colorscale = 'Temps', 
                        colorscale = 'gnbu',   
                        # colorscale = 'Tealrose',
                        # colorscale = 'Electric',
                        # colorscale = 'Plasma',
                        # colorscale = 'Inferno',    # 2022
                        showscale = True,
                        cmin = 1,
                        cmax = 6,
                        # 12, 14
                        colorbar = {'tickfont': {'size': 15, 'color': 'black', 
                                                 'family': 'Times New Roman'},
                                    'title': "Labels", 
                                    'titlefont': {'size': 18, 'color': 'black', 
                                                  'family': 'Times New Roman',
                                                  },
                                    'titleside': 'top',
                                    }
                        ),
                dimensions = list([
                    dict(range = [50.0, 51.0],
                        label = 'xNi', values = df['xNi']),
                    dict(range = [50.0, 250.0],
                        label = 'P/W', values = df['P/W']),
                    dict(range = [100.0, 1200.0],
                        label = 'v/mm/s', values = df['v/mm/s']),
                    dict(range = [70.0, 100.0],
                         label = 'h/um', values = df['h/um']),
                    dict(range = [30.0, 300.0],
                        label = 'Ev/J.mm-3', values = df['Ev/J.mm-3']),
                    dict(range = [45.0, 90.0],
                        label = 'RA', values = df['RA']),
                    dict(range = [40.0, 100.0],
                        label = 'D/um', values = df['D/um']),
                    dict(range = [100.0, 600.0],
                        label = 'Oxygen/ppm', values = df['Oxygen/ppm']),

                    dict(range = [96.0, 100.0],
                        label = 'RelativeDensity/%', values = df['RelativeDensity/%-pred']),
                    dict(range = [500.0, 800.0],
                        label = 'UTS/MPa', values = df['UTS/MPa-pred']),
                    dict(range = [5.0, 20.0],
                        label = 'EL/%', values = df['EL/%-pred']),
                    dict(range = [100.0, 320.0],
                        label = 'SIM-SMR/MPa', values = df['SIM-SMR/MPa-pred']),
                    dict(range = [260.0, 330.0],
                        label = 'Ms/K', values = df['Ms/K-pred']),
                    dict(range = [17.0, 33.0],
                        label = 'Hysteresis/K', values = df['Hysteresis/K-pred']),
                
                    dict(range = [1,  6],
                        label = 'labels', values = df['cluster_labels']),
                ]),
            # set the linewidth
            legendwidth=5,
            labelfont = {'size': 18, 'color': 'black', 'family': 'Times New Roman'},
            rangefont = {'size': 15, 'color': 'white', 'family': 'Times New Roman'},
            tickfont  = {'size': 15, 'color': 'black', 'family': 'Times New Roman'},
            )
        )

fig.update_layout(
    # title = 'Composition--Process--Properties Relationship of LPBFed NiTi SMAs', 
    title_font_family = "Times New Roman",
    title_font_color = "black",
    title_font_size = 18,
    plot_bgcolor = 'white',
    paper_bgcolor = 'white'
)


# build a folder to save the figure
save_fig_file = "parallel_coordinates"
figure = "para_coor_uts"

import os
if not os.path.exists(save_fig_file):
    os.mkdir(save_fig_file)

path = os.path.join(os.getcwd(), save_fig_file)

fig.write_html('%s/%s.html' % (path, figure))
# fig.write_html('%s/%s.png' % (path, save_fig_file)
# scope = PlotlyScope()
# with open('%s/%s.png' % (path, save_fig_file), "wb") as f:
#     f.write(scope.transform(fig, format="png"))



fig.show()