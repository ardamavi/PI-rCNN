"""
######################################################
# Main Pipeline
# by Arda Mavi
# The University of Arizona
# Department of Aerospace and Mechanical Engineering
######################################################

#### Module: "main_pipeline.py"

#### Description: Model training and plotting figures.

######################################################
"""


import os
import sys
import scipy.io
import numpy as np
from time import time
import tensorflow as tf


# Import Model Procedure:
sys.path.append("../Model")
from model_procedure import create_model, PhysicLoss, LatentLoss, get_inital_states, predict

# Import Training Procedure:
sys.path.append("../Training")
from training_procedure import training

# Import Plotting Procedure:
sys.path.append("../Plotting")
from plotting_procedure import plot_training_loss, plot_comparisons, plot_point_comparison, difference_comparison, make_gif

# Total GPU memory:
# In case of the bug occurs, reduce NVIDIA Tesla V100 GPU's 32GB memory with %70 of it for allocation:
# e.g. 'int(32000 * 0.7)' or 'None' if no GPU memory problem with full allocation.
max_gpu_memory = None

# Plot latent space:
latent_collage = True

# Plot slides as GIF:
gif_animation = True

# Main Training Plotting Pipeline:
def training_pipeline():

    # Check Devices:
    # --------------------------------------------------------------------------

    # Check CPUs:
    CPUs = tf.config.list_physical_devices('CPU')
    print("Available CPUs: ", len(CPUs))

    # Check GPUs:
    GPUs = tf.config.list_physical_devices('GPU')
    print("Available GPUs: ", len(GPUs))

    # Limit GPU Memory:
    if GPUs and max_gpu_memory:
        for i, gpu in enumerate(GPUs):
            tf.config.experimental.set_virtual_device_configuration(gpu,
            [tf.config.experimental.VirtualDeviceConfiguration( memory_limit = max_gpu_memory )])
            print("Allocated GPU:{0} memory: {1}".format(i, max_gpu_memory))



    # Parameters:
    # --------------------------------------------------------------------------
    # Train a new model OR continue with another one:
    # If new_Model = False, continue to training with the old model:
    train_new_model = True

    # If True, append new loss values to the old training records:
    # If <batch_sizes> is multi-element list, then append_loss will be
    # turned on after first training cycle:
    append_loss = False

    # Epoch size for per batch:
    epoch = 2000

    # Time step size of the training data.
    dataset_time_steps = 1000

    # Time steps:
    batch_sizes = [100]
    #############################################################################
    # If <batch_sizes> is a one element list shaped "[1000]",
    # model will be trained only with using 1000 time step.
    # If <batch_sizes> is a multi-element list shaped "[100, 200, 500, n]", then
    # model will be trained with the range indicated by the elements in the list.
    # Note: Memory limits should be considered when sellecting larger time steps for the model.
    #############################################################################

    # Learning rate:
    ## Must have same number of elemets with 'batch_sizes':
    ## Suggested: [1e-3, 9e-4, 5e-4, 4e-4] for batch_sizes [100, 200, 500, 1000]
    learning_rates = [1e-3]




    # Set Directories:
    # --------------------------------------------------------------------------
    # Main output directory
    # All outputs like checkpoints and figures will be saved into this directory:
    output_path="./Main_Outputs"

    # Checkpoint directory:
    checkpoint_dir = output_path+"/"+ "Checkpoints"
    model_folder = "Model"
    loss_file = "training_loss.npy"

    # Kernels file path:
    kernel_dir = "../PDDO_Kernels/"
    kernel_lap = "lap.mat"
    kernel_x = "G_10p.mat"
    kernel_y = "G_01p.mat"

    # Dataset file path:
    dataset_file = "../Dataset/dataset.npy"


    # Create main output directory if it is not exists:
    if not os.path.exists(output_path):
        os.makedirs(output_path)




    # Model Preparation:
    # --------------------------------------------------------------------------
    # Get model:
    Model = create_model()
    if not train_new_model:
        # Read old weights:
        Model.load_weights(checkpoint_dir+"/"+model_folder+"/")
        print("Old weights loaded.")



    # Loss Procedure Preparation:
    # --------------------------------------------------------------------------
    # Get PDDO kernels:

    kernel_lap = scipy.io.loadmat(kernel_dir+kernel_lap)["lap"]
    kernel_x = scipy.io.loadmat(kernel_dir+kernel_x)["G_10p"]
    kernel_y = scipy.io.loadmat(kernel_dir+kernel_y)["G_01p"]


    # Get loss function:
    # Main loss:
    LossFunc = PhysicLoss(kernel_lap, kernel_x, kernel_y)
    # Latent loss:
    Latent_LossFunc = LatentLoss(kernel_lap, kernel_x, kernel_y)


    # Initial Conidtion and States:
    # --------------------------------------------------------------------------
    # Get training initial conidtion from dataset:
    initial_conidtion = np.load(dataset_file, allow_pickle=True)[:1].astype("float32")

    # Get ConvLSTM's initial states:
    initial_h, initial_c = get_inital_states()

    # Initial data pack:
    initial_data = [initial_conidtion, initial_h, initial_c]



    # Training:
    # --------------------------------------------------------------------------
    # Set clock:
    t_start = time()


    # Train model:
    for batch_size, learning_rate in zip(batch_sizes, learning_rates):
        print("\n --- Time Step Size: {0} --- \n".format(batch_size))

        # Training procedure:
        Model = training(Model, initial_data, [LossFunc, Latent_LossFunc], batch_size, dataset_time_steps, epoch,
                        learning_rate, checkpoint_dir, model_folder, loss_file, append_loss)

        # Concatenate training loss for multiple time series training:
        append_loss = True


    # End clock:
    t_end = time()

    # Print Training time:
    print("Training Duration:", t_end-t_start)

    return True



################################################################################
################################################################################
################################################################################
################################################################################
################################################################################



# Main Plotting Pipeline
def plotting_pipeline():
    # Set Directories:
    # --------------------------------------------------------------------------
    # Main output directory
    # All outputs includes checkpoints and figures will be saved into this directory:
    output_path="./Main_Outputs"

    # Checkpoint directory:
    checkpoint_dir = output_path+"/"+ "Checkpoints"
    model_folder = "Model"
    loss_file = "training_loss.npy"

    # Dataset file path:
    dataset_file = "../Dataset/dataset.npy"


    # Plotting:
    # --------------------------------------------------------------------------

    # Plotting Training Loss:
    ############################################################################
    print("\n\nPlotting Figures:\n")
    # Plot training loss:
    try:
        if plot_training_loss(loss_recod_path=checkpoint_dir+"/"+loss_file, figure_path=output_path+"/"+"Figures/"):
            print("Training loss plotted.")
    except:
        print("The loss record could not be read!\nFile Path:", checkpoint_dir+"/"+loss_file)



    # Model Preparation:
    # --------------------------------------------------------------------------
    # Get model:
    Model = create_model()
    print("\nModel created.")

    # Print model architecture:
    # print(Model.summary())

    # Load trained weights:
    Model.load_weights(checkpoint_dir+"/"+model_folder+"/")
    print("\nOld weights loaded.")



    # Plotting Dataset-Model Output Comparisons:
    ############################################################################

    # Get train and test data to plot comparisons:
    dataset = np.load(dataset_file, allow_pickle=True).astype("float32")

    # Time step size that will be created by model:
    time_steps = len(dataset)-1

    # getting outputs from Model:
    print("\nModel Testing...\n")
    outputs = predict(Model, dataset[:1] , time_steps, get_latent=latent_collage)

    # Keep latent if it will be plotted:
    if latent_collage:
        output, latent = outputs
    else:
        output = outputs


    # Crop initial conidtion from dataset:
    dataset = dataset[1:time_steps+1]



    # Plot comparisons of model output and dataset:
    if plot_comparisons(ground_truths=dataset, model_outputs=output,
                        path=output_path+"/"+"Figures/Comparisons", frequency=int(time_steps/4)):
        print("\nComparison figures are saved.")


    # Plot Difference comparisons of model output and dataset:
    if difference_comparison(ground_truths=dataset[:,:,:,:1], model_outputs=output[:,:,:,:1],
                             path=output_path+"/"+"Figures/Differences_u", frequency=int(time_steps/4)):
        print("\nDifference figures are saved for u channel.")

    if difference_comparison(ground_truths=dataset[:,:,:,1:], model_outputs=output[:,:,:,1:],
                                 path=output_path+"/"+"Figures/Differences_v", frequency=int(time_steps/4)):
        print("\nDifference figures are saved for v channel.")



    # Plotting One Point Comparisons:
    ############################################################################

    # Plot train and test changes during time on a one point:
    # Choose a point (H,W) to compare:

    for point in np.random.randint(0,128, (5,2)):
        # Compare change of one point during time steps:
        if plot_point_comparison(ground_truths=dataset, model_outputs=output,
                                 comparison_point=point, path=output_path+"/"+"Figures/", file="point_comparison{0}x{1}.png".format(point[0], point[1])):
            print("\nPoint comparison figure saved.")



    # Plot Latent Collage:
    ############################################################################
    if latent_collage:
        # Create collage to make GIF
        frame = np.sqrt(latent.shape[-1])
        h = int(frame*latent.shape[1])
        w = int(frame*latent.shape[2])
        collage = np.zeros((latent.shape[0],h,w))
        for t in range(latent.shape[0]):
            for i_h, c_h in enumerate(range(0, h, latent.shape[1])):
                for i_w, c_w in enumerate(range(0, h, latent.shape[2])):
                    collage[t, c_h:c_h+latent.shape[1], c_w:c_w+latent.shape[2]] = latent[t, :, :, int((i_h*frame)+i_w)]
        collage[:,h//2,:] = 1. # Line seperation for U and V channels
        # Make GIF:
        make_gif(data=collage, path=output_path+"/"+"Figures/GIFs/", file="latent.gif", min_val=-1., max_val=1.)

        print("\nLatent collage saved.")



    # GIF Animations:
    ############################################################################
    if gif_animation:

        # Make GIF animation:
        # Dataset:
        # Channel u:
        make_gif(data=dataset[:,:,:,0], path=output_path+"/"+"Figures/GIFs/", file="dataset_u.gif")
        # Channel v:
        make_gif(data=dataset[:,:,:,1], path=output_path+"/"+"Figures/GIFs/", file="dataset_v.gif")

        # Model Outputs:
        # Channel u:
        make_gif(data=output[:,:,:,0], path=output_path+"/"+"Figures/GIFs/", file="model_outputs_u.gif")
        # Channel v:
        make_gif(data=output[:,:,:,1], path=output_path+"/"+"Figures/GIFs/", file="model_outputs_v.gif")

        print("\nGIFs saved.")

    return True

    return True



# # Main pipeline:
if __name__ == "__main__":

    # Choosing modes:
    mode_choosing = """\n
    '-h' OR '-help' Help descriptions.
    '-m train'  ->  Only runs training pipleline. \n
    '-m plot'  ->  Only runs plotting pipleline. \n
    If there is no argument specified, runs both training and plotting pipeline. \n
    If there is a wrong argument given, help description will show up. \n
    """

    # Check arguments:
    args = sys.argv
    if "-m" in args:
        if "train" in args:
            # Training:
            print("Training Procedure:")
            training_pipeline()
        if "plot" in args:
            # Plotting:
            print("Plotting Procedure:")
            plotting_pipeline()
    elif "-h" in args or "help" in args or len(args) > 1:
        print(mode_choosing)
    elif len(args) == 1:
        # Training:
        print("Training Procedure:")
        training_pipeline()

        # Plotting:
        print("Plotting Procedure:")
        plotting_pipeline()
