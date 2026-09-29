import matplotlib
from matplotlib import font_manager
import os
import numpy as np
import matplotlib.patches as patches

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Inspired by Griffin Chure

def matplotlib_style(return_colors=True, return_palette=True, **kwargs):
    """
    Assigns the plotting style for matplotlib generated figures. 
    
    Parameters
    ----------
    return_colors : bool
        If True, a dictionary of the colors is returned. Default is True.
    return_palette: bool
        If True, a sequential color palette is returned. Default is True.
    """

    # Get font
    
    font_path = "/Users/tomroeschinger/Library/Fonts/Lucida-Sans-Unicode-Regular.ttf" 
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)

    # Define the matplotlib styles.
    rc = {
        # Axes formatting
        "axes.facecolor": "#f7f7fa",
        "axes.edgecolor": "#ffffff",
        "axes.labelcolor": "#000000",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.axisbelow": True,
        "axes.linewidth": 0.15,
        "axes.grid": True,

        # Formatting of lines and points. 
        "lines.linewidth": 0.5,
        "lines.dash_capstyle": "butt",
        "patch.linewidth": 0.25,
        "lines.markeredgecolor": '#ffffff',
        "lines.markeredgewidth": 0.5,

        # Grid formatting
        "grid.linestyle": ':',
        "grid.linewidth": 0.25,
        "grid.color": '#FFF',

        # Title formatting
        "axes.titlesize": 8,
        "axes.titleweight": 700,
        "axes.titlepad": 3,
        "axes.titlelocation": "center",

        # Axes label formatting. 
        "axes.labelpad": 2,
        "axes.labelweight": 700,
        "xaxis.labellocation": "center",
        "yaxis.labellocation": "center",
        "axes.labelsize": 8,
        "axes.xmargin": 0.03,
        "axes.ymargin": 0.03,

        # Legend formatting
        "legend.fontsize": 5,
        "legend.labelspacing": 0.25,
        "legend.title_fontsize": 5,
        "legend.frameon": True,
        "legend.edgecolor": "#5b5b5b",

        # Tick formatting
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.width": 0.25,
        "ytick.major.width": 0.25,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,

        # General Font styling
        "font.weight": 400, # Weight of all fonts unless overriden.
        "font.style": "normal",
        "text.color": "#000000",

        # Higher-order things
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    if os.path.exists(font_path):
        rc["font.family"] = "sans-serif",
        rc["font.sans-serif"] = prop.get_name()
    matplotlib.style.use(rc)

# plotting parameters
def notebook_params():
    rc = {
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xlabel.fontsize': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'axes.grid': True
    }
    matplotlib.style.use(rc)

def paper_params():
    rc = {
        'xtick.labelsize': 3,
        'ytick.labelsize': 3,
        'axes.titlesize': 6,
        'axes.labelsize': 6,
        'figure.labelsize': 6,
        'axes.grid': True,
        'axes.linewidth': 0.2,
        
        'xtick.major.size': 1,  # Major tick length
        'xtick.major.width': 0.5,   # Major tick width
        'ytick.major.size': 1,
        'ytick.major.width': 0.5,
        
        'xtick.minor.size': .5,  # Minor tick length
        'xtick.minor.width': .1,   # Minor tick width
        'ytick.minor.size': .5,
        'ytick.minor.width': .1}

    matplotlib.style.use(rc)

def get_pixels(points, matplotlib_dpi=100):
    illustrator_dpi = 72
    
    return points/illustrator_dpi * matplotlib_dpi 
    
my_color_dict = {
    "orange1" : "#f47c20",
    "orange2" : "#fecc96",
    "orange3" : "#ffe4c6",
    "yellow1" : "#fce317",
    "yellow2" : "#fff182",
    "yellow3" : "#fff8c1",
    "green1" : "#a8cf38",
    "green2" : "#d1e39b",
    "green3" : "#e6f0cb",
    "blue1" : "#324fa2",
    "blue2" : "#8d92c8",
    "blue3" : "#dbddef",
    "purple1" : "#9f2260",
    "purple2" : "#cca6b6",
    "purple3" : "#e9d1da",
    "red1" : "#D14241",
    "red2" : "#E59C8C",
    "red3" : "#F0CABF",
    "gray1": "#474747",
    "gray2": "#7e7e7e", 
    "gray3": "#bdbdbd",
}



def plot_exshift(expression_shift, xscale=1, position_offset=-115):
    colors_with_points = [
            (0, "#63ACBE"),  
            (0.1, "#63ACBE"),  
            (.5, "#ffffff"),    
            (0.9, "#EF875F"),
            (1, "#EF875F")
        ]
    
        # Build the colormap
    my_cmap = LinearSegmentedColormap.from_list("my_cmap",colors_with_points)
    # data dimensions
    
    rows, cols = expression_shift.shape

    # Determine symmetric normalization centered at zero
    max_abs_val = np.max(np.abs(expression_shift))
    norm = Normalize(vmin=-max_abs_val, vmax=max_abs_val)
    # set figure size proportional to data dimensions
    fig_width = cols * 0.2/xscale  # adjust 0.2 as needed for scaling
    fig_height = rows * 0.2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

   # Iterate and draw rectangles colored by expression_shift
    for row in np.arange(rows):
        for col in np.arange(cols):
        # Normalize the value to [0,1] for colormap
            norm_value = norm(expression_shift[row, col])
            color = my_cmap(norm_value)
            
            rect = patches.Rectangle(
                ((col+position_offset)/xscale-0.5/xscale, row-0.5), 1/xscale, 1,
                linewidth=0, facecolor=color
            )
            ax.add_patch(rect)

    # Set axis limits
    ax.set_xlim(-115.5/xscale, 44.5/xscale)
    ax.set_ylim(-0.5, 3.5)

    # Optional: remove axes for cleaner visualization
    #ax.axis('off')

    width_pts = 100  
    height_pts = 40
    dpi = 100
    
    # Calculate figsize in inches
    width_in = get_pixels(width_pts) / dpi
    height_in = get_pixels(height_pts) / dpi
    #fig.set_size_inches(width_in, height_in)
    ax.set_yticks([0, 1, 2, 3], ['A', 'C', 'G', 'T'], fontsize=2)
    ax.set_xticks(np.arange(-115, 45, step=10)/xscale, np.arange(-115, 45, step=10), fontsize=2)
    ax.grid(False)
    return fig, ax