"""
######################################################
# Plotting Procedure
# by Arda Mavi
# The University of Arizona
# Department of Aerospace and Mechanical Engineering
######################################################

#### Module: "plotting_procedure.py"

#### Description: Creates figues of train and testing outputs, and training loss.

#### Functions:
#---------------------------------------------------------

Func: plot_training_loss(loss_recod_path="../Training/Checkpoints/training_loss.npy",
                plot_file="Training_Loss.png", figure_path="Figures/")

Args:
    - loss_recod_path (String):
        Default: "../Training/Checkpoints/training_loss.npy"
        Training loss recods as Numpy file that created during training.
    - plot_file (String):
        Default: "Training_Loss.png"
        File name of the figure.
    - figure_path (String):
        Default: "Figures/"
        Directory to save figure.

Returns:
    - Bool (Boolean):
        If True, plotting completed successfully.

#---------------------------------------------------------

Func: plot_comparisons(ground_truths, model_outputs, path="Figures/Comparison", frequency=100)

Args:
    - ground_truths (Numpy Array):
        Model output data (y) with shape (t,h,w,c).
        Must be same shape with <model_outputs>.
    - model_outputs (Numpy Array):
        Model output data (y') with shape (t,h,w,c).
        Must be same shape with <ground_truths>.
    - path (String):
        Default: "Figures/Comparison"
        Folder path to save comparison figures.
    - frequency (Int):
        Default: 100
        Time step range gap of data comparison figures.

Returns:
    - Bool (Boolean):
        If True, plottings completed successfully.

#---------------------------------------------------------

Func: plot_point_comparison(ground_truths, model_outputs, comparison_point=(32,32),
                            path="Figures/", file="point_comparison.png", time_interval=3)

Args:
    - ground_truths (Numpy Array):
        Model output data (y) with shape (t,h,w,c).
        Must be same shape with <model_outputs>.
    - model_outputs (Numpy Array):
        Model output data (y') with shape (t,h,w,c).
        Must be same shape with <ground_truths>.
    - comparison_point (Tuple):
        Default: (32,32)
        (H, W) shaped tuple that represent selected pixel point coordinate.
    - path (String):
        Default: "Figures/"
        Folder path to save point comparison figure.
    - file (String):
        Default: "point_comparison.png"
        File name of point comparison figure.
    - time_interval (Int or Float):
        Default: 3 (Second)
        Time interval of the <ground_truths> and <model_outputs>.

Returns:
    - Bool (Boolean):
        If True, plotting completed successfully.

#---------------------------------------------------------

Func: difference_comparison(ground_truths, model_outputs, path="Figures/Differences", frequency=100)

Args:
    - ground_truths (Numpy Array):
        Model output data (y) with shape (t,h,w,1).
        Must be same shape with <model_outputs>.
    - model_outputs (Numpy Array):
        Model output data (y') with shape (t,h,w,1).
        Must be same shape with <ground_truths>.
    - path (String):
        Default: "Figures/Differences"
        Folder path to save difference figures.
    - frequency (Int):
        Default: 100
        Time step range gap of difference figures.

Returns:
    - Bool (Boolean):
        If True, plotting completed successfully.

#---------------------------------------------------------

Func: make_gif(data, path="GIFs/", file="data.gif", duration=0.02, min_val=-1., max_val=1.)

Args:
    - data (Numpy Array):
        3D Numpy array that holds data values with shape (time, high, width).
    - path (String):
        Default: "GIFs/"
        Path of saving directory.
    - file (String)
        Default: "data.gif"
        File name of gif.
    - duration (Float):
        Default: 0.02
        Duration between frames of the GIF.
    - min_val, max_val (Float):
        Default: min_val=-1., max_val=1.
        Trims higher and lower values in the <data>.

Returns:
    - Bool (Boolean):
        If True, GIF saved successfully.

#---------------------------------------------------------
"""


import os
import numpy as np
from imageio import mimsave
from matplotlib import pyplot as plt


# Plot training loss:
def plot_training_loss(loss_recod_path="../Training/Checkpoints/training_loss.npy",
              plot_file="Training_Loss.png", figure_path="Figures/"):

    # Create folder if not exists:
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    # Get training loss:
    loss_recod = np.load(loss_recod_path)

    # Get x axis that represents training epochs:
    x_axis = list(range(1, len(loss_recod)+1))

    # Plotting training loss:
    plt.plot(x_axis, loss_recod)
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim([ 1, len(loss_recod) ])
    plt.ylim([ loss_recod.min(), np.median(loss_recod)*2 ]) # Window

    # Save plot:
    plt.savefig(figure_path+"/"+plot_file)
    plt.close("all")

    return True



# Plot comparisons of dataset and model outputs:
def plot_comparisons(ground_truths, model_outputs, path="Figures/Comparison", frequency=100):

    # Create folder if not exists:
    if not os.path.exists(path):
        os.makedirs(path)

    # Get data shape:
    data_shape = model_outputs.shape

    # Get grid:
    x = np.linspace(0, 1, data_shape[2])
    y = np.linspace(0, 1, data_shape[1])
    grid_x, grid_y = np.meshgrid(x,y)

    # Plot data with <frequency> gap:
    for index in range(0, data_shape[0]+1, frequency):
        # Get couples:
        index = abs(index-1) # Range: [1,+inf)
        y_dataset = ground_truths[index]
        y_model = model_outputs[index]

        # Plotting:
        title_list = [["Reference u", "Reference v",],
                      ["Model Output u", "Model Output v"]]
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
        fig.subplots_adjust(hspace=0.25, wspace=0.25)
        for plt_y in range(0,2):
            for plt_x in range(0,2):
                data = y_dataset if plt_y == 0 else y_model # Specify data to plot
                p = ax[plt_y, plt_x].scatter(grid_x, grid_y, s=1, c=data[:,:,plt_x], marker="s",
                    cmap="coolwarm", vmin=-0.7, vmax=0.7, alpha=1, edgecolors=None)
                ax[plt_y, plt_x].axis("square")
                ax[plt_y, plt_x].set_title(title_list[plt_y][plt_x])

        # Draw ColorBar:
        # Left, bottom, width, and height of the Color Bar:
        color_bar_coord = fig.add_axes([0.915, 0.3, 0.01, 0.4])
        fig.colorbar(p, cax=color_bar_coord, orientation="vertical")

        # Save plot:
        plt.savefig(path+"/Comparisons_t{0}.png".format(index+1))
        plt.close("all")

    return True



# Plot the comparisons of changes on one point during time:
def plot_point_comparison(ground_truths, model_outputs, comparison_point=(32,32),
                          path="Figures/", file="point_comparison.png", time_interval=3):

    # Create folder if not exists:
    if not os.path.exists(path):
        os.makedirs(path)

    # Get x axis that represents time:
    x_axis = np.linspace(0, time_interval, len(model_outputs))

    # Get changes on one point:
    dataset_change = ground_truths[:,comparison_point[0], comparison_point[1], :]
    model_change = model_outputs[:,comparison_point[0], comparison_point[1], :]


    # Plotting:
    title_list = ["u change on point ({0},{1})".format(comparison_point[0], comparison_point[1]),
                  "v change on point ({0},{1})".format(comparison_point[0], comparison_point[1])]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 6))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    for plt_x in range(0,2):
        c = 0 if plt_x == 0 else 1 # Specify channel to plot
        ax[plt_x].plot(x_axis, dataset_change[:,c], "tab:green", label="Reference")
        ax[plt_x].plot(x_axis, model_change[:,c], "tab:orange", label="Model")
        ax[plt_x].set_title(title_list[plt_x])
        ax[plt_x].set_xlabel('Time')
        ax[plt_x].set_ylabel("u" if c == 0 else "v")
        ax[plt_x].set_xlim([0, time_interval])
        ax[plt_x].legend()

    # Save plot:
    plt.savefig(path+"/"+file)
    plt.close("all")

    return True



# Difference Comparison:
def difference_comparison(ground_truths, model_outputs, path="Figures/Differences", frequency=100):

    # Create folder if not exists:
    if not os.path.exists(path):
        os.makedirs(path)

    # Get data shape:
    data_shape = model_outputs.shape

    # Plot data with <frequency> gap:
    for index in range(0, data_shape[0]+1, frequency):
        # Get couples:
        index = abs(index-1) # Range: [1,+inf)
        y_dataset = ground_truths[index]
        y_model = model_outputs[index]

        data = y_dataset - y_model
        min_val = data.min()
        max_val = data.max()

        # Plotting:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        fig.subplots_adjust(hspace=0.25, wspace=0.25)
        p = ax.imshow(data[:,:,0], cmap="coolwarm", vmin=min_val, vmax=max_val)
        ax.set_xlim([0, data_shape[1]])
        ax.set_ylim([0, data_shape[2]])
        ax.axis("square")
        ax.set_title("Difference")

        # Draw ColorBar:
        # Left, bottom, width, and height of the Color Bar:
        color_bar_coord = fig.add_axes([0.915, 0.3, 0.01, 0.4])
        plt.colorbar(p, cax=color_bar_coord, orientation="vertical")

        # Save plot:
        plt.savefig(path+"/Difference_t{0}.png".format(index+1))
        plt.close("all")

    return True




# Creates GIF Animations:
def make_gif(data, path="GIFs/", file="data.gif", duration=0.02, min_val=-1., max_val=1.):

    # Create folder if not exists:
    if not os.path.exists(path):
        os.makedirs(path)

    # Cut the max and min points:
    data[data > max_val] = max_val
    data[data < min_val] = min_val

    # Normalize between [0, 1]:
    data += abs(data.min())
    data /= data.max()

    # Get color map from data:
    getColorMap = plt.get_cmap("coolwarm")
    data = getColorMap(data)

    # Float [0,1] to Int [0,255]
    data = (data*255).astype(np.uint8)

    # Write GIF:
    mimsave(path+"/"+file, data, format='GIF', duration=duration)

    return True
