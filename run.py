import argparse
import yaml
import os

from train_utils import train
from write_results import *
from utils import *

def get_args():
    parser = argparse.ArgumentParser(description='CV with selected experiment as test set and train/val (+test) stratified from the others')
    parser.add_argument("-c", "--config_path", type=Path, help="Path to the config.", default="configs/baseline.yaml")
    parser.add_argument("-d", "--dataset", type=str, help="Select dataset version from wandb Artifact (v1, v2...), set to 'nw' (no WB) to use paths from the config file. Default is 'raw'.", default='raw')
    parser.add_argument("--tag", type=str, help="Add custom tag on the wandb run (only one tag is supported).", default=None)
    parser.add_argument("-e", "--exp", default=None, type=int, help="Experiment to put in test set")
    parser.add_argument("--single_exp", default=None, type=int, help="Perform CV only on a single experiment.")
    parser.add_argument("-t", "--tube", default=None, type=int, help="If present, select a single tube as test set (integer index between 0 and 31).")
    parser.add_argument("--noG_preprocessing", nargs='?', type=str2bool, default=False, const=True, help="If applied, uses the no_G preprocessing.")
    
    parser.add_argument("-m", "--model", type=str, default=None, help="Select model different from U++.")
    parser.add_argument('-l', '--loss', type=str, default=None, help="Select loss function, default is combo Focal+Jaccard ('base').")
    parser.add_argument("-k", "--k", type=int, default=0, help="Number of the fold to consider between 0 and 4.")
    parser.add_argument("-s", "--seed", type=int, default=7, help="Change the seed to the desired one.")
    
    parser.add_argument("--stratify_fold", nargs='?', type=str2bool, default=False, const=True, help="Split dataset with StratifiedKFold instead of GroupKFold.")
    
    parser.add_argument("-f", "--focus_size", default=None, help="Select 'small_cysts' ('s') or 'big_cysts' ('b') labels (only avaiable from 'v0' dataset).")
    parser.add_argument("--tiling", nargs='?', type=str2bool, default=False, const=True, help="If applied, uses the latest tiled-dataset available in WB.")
    
    parser.add_argument('--save_results', nargs='?', type=str2bool, default=False, const=True, help = "Prevent Wandb to save validation result for each step.")
    parser.add_argument('--wb', nargs='?', type=str2bool, default=False, const=True, help = "Prevent Wandb to save validation result for each step.")
    parser.add_argument('--evaluate_exp', nargs='?', type=str2bool, default=False, const=True, help = "Apply the evaluation strategy at the end of the train/test pipeline.")
    
    return parser.parse_args()

def main(args):
    if args.model in ["segformer", "esfpnet"]: 
        print("Model not implemented")
        if wandb.run: wandb.finish()
        return

    os.environ["WANDB_START_METHOD"] = "fork"
    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    name = "crossval_" + "_".join([f"{hp}_{getattr(args, hp)}" for hp in ["model", "loss", "tube", "exp", "noG_preprocessing", "seed"] if getattr(args, hp, None) is not None])
    
    torch.set_float32_matmul_precision('medium')
    train(args, hparams, name)
    
    if args.evaluate_exp:
        evaluator = Evaluator(
            predP=hparams["checkpoint_callback"]["dirpath"]/'result'/'test',
            imgP=hparams["image_path"], maskP=hparams["mask_path"]
            )
        evaluator.write_results()

    if wandb.run is not None:
        wandb.finish()
    return


if __name__ == '__main__':
    args = get_args()
    main(args)
