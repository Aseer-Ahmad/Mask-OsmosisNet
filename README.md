## BSDS300 dataset

BSDS300 dataset is uploaded at the following. Running the training code will automatically download and process it.
```
https://huggingface.co/datasets/aseeransari/BSDS
```

# Setting up the environment
This code works with python 3.10. Further to install the dependecies for python code, run the following 

To install pytorch (windows and linux) run: 

pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121

Then run

```
pip install -r req.txt
```

# Configuration file

The [config.yaml](config.yaml) file contains all the configurable paramters for surrogate and numerical network models. 

```
EPOCHS : 100 
MODEL_TYPE : unet_standard <SETS THE MODEL ARCHITECTURE>
RESUME_CHECKPOINT : <MODEL CHECKPOINT FILE TO RESUME TRAINING FROM>
SAVE_EVERY_ITER : 25 <ITERATION TO SAVE MODEL CHECKPOINT >
VAL_EVERY_ITER : 25 <ITERATION TO VALIDATION MODEL CHECKPOINT >
BATCH_PLOT_EVERY_ITER : 2 <ITERATION TO PLOT RESUTS OF MODEL >
OUTPUT_DIR : outputs <OUTPUT DIRECTORY NAME>
EXP_NAME : 27_4_SM_1 <EXPERIMENT NAME TO PUT INSIDE OUTPUT_DIR FOLDER>

IMG_SIZE       : 128 <IMAGE RESIZE RESOLUTION>
INP_CHANNELS   : 1   <INPUT CHANNEL TO MODEL>
OUT_CHANNELS   : 2   <OUTPUT CHANNEL TO MODEL>

# HYPER PARAMS unet
OPT            : 
    opt        : Adam  <OPTIMIZER SELECTED. Options : SGD-Nestrov, SGD, AdamW, RMSprop, Adam>
    lr         : 0.0001 <LEARNING RATE OF OPTIMIZER>
    wd         : 0.001 <WEIGHT DECAY OF OPTIMIZER>
    momentum   : 0.009   <MOMENTUM OF OPTIMZER>

SCHEDL         : 
TRAIN_BATCH    : 4 <SIZE OF TRAIN BATCH>
TEST_BATCH     : 4 <SIZE OF TEST BATCH>
ALPHA1         : 0.01 <REGULARIZATION PARAMTER ALPHA FOR MASK LOSS>
MASK_DEN       : 0.1   <TARGET MASK DENSITY>
MAX_NORM       : 1      <MAX NORM TO SET FOR GRADIENT CLIPPING>
SKIP_NORM      : 500    <MAX NORM ABOVE WHICH TO SKIP AND RESCALE GRADIENTS>

SOLVER_TYPE      : Stab_BiCGSTAB <TYPE OF OSMOSIS SOLVER. Options : Stab_BiCGSTAB, Jacobi>
```

# Running neural network training

To train the joint network with surrogate solver, make sure joint model parameters are correctly set and then run 

```
python main_joint.py
```

To train the single mask network with numerical solver, set OUT_CHANNELS in config.yaml to 1 and then run

```
python main.py
```

To train the single mask network with numerical solver, set OUT_CHANNELS in config.yaml to 2 and then run

```
python main.py
```

The ouputs are written to OUTPUT_DIR/EXP_NAME based on [config.yaml.](config.yaml).

# Inference

For inference, with trained model, follow the directory structure of folder named Inference. It contains a file called [infer.yaml](Inference/infer.yaml) that contains paramters for model checkpoint, dataset etc. It also contains all the results already computed for BSDS300 dataset. The current [infer.yaml ](Inference/infer.yaml) contains the parameters set for all three trained models used in the thesis.  

The following folders contain : 

```
Inference/masknet_wts     : weights for mask network of the joint surrogate network. 
Inference/unet_double_wts : weights for mask network of double mask numerical solver  network.
Inference/unet_single_wts : weights for mask network of single mask numerical solver  network.
Inference/outputs         : results for BSDS300 dataset by model type. folder name can be understood as 
    - single : single mask joint surrogate solver model
    - double : double mask joint surrogate solver model
    - masknet : mask network for numerical solver model
```

For inference dataset, data need to put inside [dataset/BSDS_extras/images/test](dataset/BSDS_extras/images/test) with a file [dataset/BSDS_extras/images/iids_test.txt](dataset/BSDS_extras/images/iids_test.txt) that contains names of all the images. 

Finally to run the inference code, run 

```
python infer.py
```

This generates results based on the tasks listed in [infer.yaml](Inference/infer.yaml) and a .csv file with metrics recorded in it.

# C code 

Before running the code, make sure to have a directory named [outputs](MaskSelection/outputs) in the code folder. Intermediate results for sparsification will be written to this. 

```
MaskSelection/belhachmi.c            : contains code for analytic approach of belhachmi
MaskSelection/prob-spars-os-global.c : contains code for probabalistic sparsification with global error criteria for osmosis
MaskSelection/prob-spars-os-local.c  : contains code for probabalistic sparsification with local error criteria for osmosis
MaskSelection/osmosis_guidance_image_im.c : contains code for osmosis with drift vector modification
```

The compiled filed for the codes are the following : 

```
MaskSelection/belhachmi.out
MaskSelection/ps_global.out
MaskSelection/ps_local.out
MaskSelection/osmosis.out
```

MaskSelection/belhachmi.out requires command line paramaters. Example : 

```
/belhachmi.out -i <INPUT_FILE> -m <MASK_FILE> -d 0.1 -I
```

To recompile the code use, 
```
gcc -Wall -O2 -o <OUT_FILE_NAME> <C_FILENAME> -lm
```
