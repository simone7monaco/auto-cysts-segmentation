import argparse
import torch
from pathlib import Path

import pydoc
from typing import Any, Union, Dict, List, Tuple,Optional
from zipfile import ZipFile
from easydict import EasyDict as ed
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, GroupKFold
import cv2
import re

import numpy as np
import torch
import wandb


def get_id2_file_paths(path: Union[str, Path]) -> Dict[str, Path]:
    return {x.stem: x for x in Path(path).glob("*.*")}



def date_to_exp(date):
    date_exps = {'0919': 1, '1019': 2, '072020':3, '092020':3, '122020':4, '0721':5}
    date = ''.join((date).split('.')[1:])
    return date_exps[date]


# TODO: togliere
# all_treats = {'ctrl', 'treat_1', 'treat_2', 'treat_3', 'treat_4', 'treat_5', 'treat_6', 'treat_7', 'treat_8', 'treat_9', 'treat_10'}
all_treats = {'ctrl', 't3', 'triac', 't4', 'tetrac', 'resv', 'dbd', 'lm609', 'uo', 'dbd+t4', 'uo+t4', 'lm609+t4', 'lm609+10ug.ml', 'lm609+2.5ug.ml'}

def simplify_names(filename):
    unpack = re.split(' {1,}_?|_', filename.strip())
    
    date_idx = [i for i, item in enumerate(unpack) if re.search('[0-9]{1,2}.[0-9]{1,2}.[0-9]{2,4}', item)][0]
    unpack = unpack[date_idx:]
    date = unpack[0]
    treatment = [x.upper() for x in unpack if x.lower() in all_treats][-1]

    side = [s for s in unpack if re.match('A|B', s)]
    side = side[0] if side else 'U'

    zstack = [s.lower() for s in unpack if re.match('[24]0[xX][0-9]{1,2}', s)][0]
    alt_zstack = [s for s in unpack if re.match('\([0-9]{1,}\)', s)]
    if alt_zstack: zstack = zstack.split('x')[0] + 'x' + alt_zstack[0][1:-1]
    z1, z2 = zstack.split('x')
    zstack = f"{z1}x{int(z2):02}"

    tube = [n for n in unpack if re.fullmatch('[0-9]*', n)][0]
    
    return date, treatment, tube, zstack, side


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_samples(image_path: Path, mask_path: Path) -> List[Tuple[Path, Path]]:
    """Couple masks and images.

    Args:
        image_path:
        mask_path:

    Returns:
    """

    image2path = get_id2_file_paths(image_path)
    mask2path = get_id2_file_paths(mask_path)

    return [(image_file_path, mask2path[file_id]) for file_id, image_file_path in image2path.items()]


def str_from_samples(samples, onlyname=False):
    if not samples: return []
    extr = lambda st: st.name if onlyname else str(st)
    return [(extr(p[0]), extr(p[1])) for p in samples]


def find_average(outputs: List, name: str) -> torch.Tensor:
    if len(outputs[0][name].shape) == 0:
        return torch.stack([x[name] for x in outputs]).mean()
    return torch.cat([x[name] for x in outputs]).mean()


def rename_layers(state_dict: Dict[str, Any], rename_in_layers: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in state_dict.items():
        for key_r, value_r in rename_in_layers.items():
            key = re.sub(key_r, value_r, key)

        result[key] = value

    return result


def state_dict_from_disk(
    file_path: Union[Path, str], rename_in_layers: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Loads PyTorch checkpoint from disk, optionally renaming layer names.
    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}
    Returns:
    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if rename_in_layers is not None:
        state_dict = rename_layers(state_dict, rename_in_layers)

    return state_dict


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034
    return pydoc.locate(object_type)(**kwargs) if pydoc.locate(object_type) is not None else pydoc.locate(object_type.rsplit('.', 1)[0])(**kwargs)


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(image)


def load_rgb(image_path: Union[Path, str]) -> np.array:
    """Load RGB image from path.
    Args:
        image_path: path to image
        lib: library used to read an image.
            currently supported `cv2` and `jpeg4py`
    Returns: 3 channel array with RGB image
    """
    if Path(image_path).is_file():
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    raise FileNotFoundError(f"File not found {image_path}")


def load_mask(path):
    im = str(path)
    return cv2.imread(im, cv2.IMREAD_GRAYSCALE)


def binary_mean_iou(logits: torch.Tensor, targets: torch.Tensor, EPSILON = 1e-15) -> torch.Tensor:
    output = (logits > 0.5).int()

    if output.shape != targets.shape:
        targets = torch.squeeze(targets, 1)

    intersection = (targets * output).sum()
    union = targets.sum() + output.sum() - intersection
    result = (intersection + EPSILON) / (union + EPSILON)

    return result


#######################
####                  #
#### Training utils   #
####                  #
#######################

def init_training(args, hparams, name, tiling=False):
    """
    Variable "args" is supposed to be the argparse output, while the hparams is the dictionary
    from the configuration file, which will be eventually updated
    """
    
    if type(args) == dict:
        args = ed(args)

    if args.dataset != 'nw' and args.wb and not tiling:
        dataset = wandb.run.use_artifact(f'smonaco/rene-policistico-artifacts/dataset:{args.dataset}', type='dataset')
        data_dir = dataset.download()

        if not (Path(data_dir) / "images").exists():
            zippath = next(Path(data_dir).iterdir())
            with ZipFile(zippath, 'r') as zip_ref:
                zip_ref.extractall(data_dir)

        hparams["image_path"] = Path(data_dir) / "images"
        hparams["mask_path"] = Path(data_dir) / "masks"
    elif tiling:
        data_dir = "artifacts/tiled-dataset:v0"
        hparams["image_path"] = Path(data_dir) / "images"
        hparams["mask_path"] = Path(data_dir) / "masks"
    else:
        hparams["image_path"] = Path(hparams["image_path"])
        hparams["mask_path"] = Path(hparams["mask_path"])
    
    if hasattr(args, 'use_scheduler') and not args.use_scheduler:
        hparams['scheduler'] = None

    hparams["checkpoint_callback"]["dirpath"] = Path(hparams["checkpoint_callback"]["dirpath"])
    hparams["checkpoint_callback"]["dirpath"] /= name
    hparams["checkpoint_callback"]["dirpath"].mkdir(exist_ok=True, parents=True)

    if hasattr(args, 'noG_preprocessing'):
        hparams["noG_preprocessing"] = args.noG_preprocessing
    
    hparams['seed'] = args.seed
    hparams['tube'] = args.tube

    print("---------------------------------------")
    print("        Running Crossvalidation        ")
    if args.tiling:
        print("         with tiled dataset        ")
    if args.model is not None:
        print(f"         model: {args.model}  ")
    if args.exp is not None:
        print(f"           exp: {args.exp}  ")
    if args.tube is not None:
        print(f"         tube: {args.tube}  ")
    print(f"         seed: {args.seed}           ")
    print(f"         fold: {args.k}       ")
    print("---------------------------------------\n")
    
    return hparams



def split_dataset(hparams):
    samples = get_samples(hparams["image_path"], hparams["mask_path"])
    k=getattr(hparams, 'k', 0)
    test_exp=getattr(hparams, 'exp', None)
    leave_one_out=getattr(hparams, 'tube', None)
    strat_nogroups=getattr(hparams, 'stratify_fold', None)
    single_exp=getattr(hparams, 'single_exp', None)
    
    ##########################################################
    if single_exp == 1:
        samples = [u for u in samples if "09.19" in u[0].stem]
    if single_exp == 2:
        samples = [u for u in samples if "10.19" in u[0].stem]
    if single_exp == 3:
        samples = [u for u in samples if "07.2020" in u[0].stem or "09.2020" in u[0].stem]
    if single_exp == 4:
        samples = [u for u in samples if "12.2020" in u[0].stem]
#         samples = [u for u in samples if "ctrl 11" in u[0].stem.lower() or "t4" in u[0].stem.lower()]
    if single_exp == 5:
        samples = [u for u in samples if "07.21" in u[0].stem]
    ##########################################################
    
    names = [file[0].stem for file in samples]

    unpack = [simplify_names(name) for name in names]
    df = pd.DataFrame({
        "filename": names,
        "treatment": [u[1] for u in unpack],
        "exp": [date_to_exp(u[0]) for u in unpack],
        "tube": [u[2] for u in unpack],
    })
#     df["te"] = df.treatment + '_' + df.exp.astype(str)
    df["te"] = (df.treatment + '_' + df.exp.astype(str) + '_' + df.tube.astype(str)).astype('category')
    
    if test_exp is not None or leave_one_out is not None:
        if leave_one_out is not None:
            tubes = df[['exp','tube']].astype(int).sort_values(by=['exp', 'tube']).drop_duplicates().reset_index(drop=True).xs(leave_one_out)
            test_idx = df[(df.exp == tubes.exp)&(df.tube == str(tubes.tube))].index     
        else:
            test_idx = df[df.exp == test_exp].index
    
        test_samp = [x for i, x in enumerate(samples) if i in test_idx]
        samples = [x for i, x in enumerate(samples) if i not in test_idx]
        df = df.drop(test_idx)
    else:
        test_samp = None
        
    if strat_nogroups:
        skf = StratifiedKFold(n_splits=5, random_state=hparams["seed"], shuffle=True)
        train_idx, val_idx = list(skf.split(df.filename, df.te))[k]
    else:
        df, samples = shuffle(df, samples, random_state=hparams["seed"])
        gkf = GroupKFold(n_splits=5)
        train_idx, val_idx = list(gkf.split(df.filename, groups=df.te))[k]
    
    train_samp = [tuple(x) for x in np.array(samples)[train_idx]]
    val_samp = [tuple(x) for x in np.array(samples)[val_idx]]
    
    return {
        "train": train_samp,
        "valid": val_samp,
        "test": test_samp
    }
