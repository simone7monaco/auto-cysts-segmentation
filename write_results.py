import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, confusion_matrix
import argparse

from utils import simplify_names, load_mask, load_rgb

def get_args():
    parser = argparse.ArgumentParser(description='CV with selected experiment as test set and train/val stratified from the others')
    parser.add_argument("-i", "--inpath", type=Path, help="Path containing image predicitions.", required=True)
    parser.add_argument("-d", "--dataset", default=None)
    
    return parser.parse_args()


# 3 µm (valore minimo) ad uno di circa 40 µm (valore massimo). I valori medi invece sono tra i 6 e i 13 µm
# 25 : 72 = µm : px
to_px = lambda m : m * 72 / 25
to_m = lambda px : px * 25 / 72
to_area_m = lambda Apx : (25/72)**2 * Apx
area_mc = lambda Apx : (25/72)**2 * Apx * 1e-6 ## to mm²

circ_area = lambda d : np.pi * (d/2)**2

min_area = circ_area(3)
max_area = circ_area(40)
mid_areas = circ_area(np.array([6, 13]))




def unpack_name(name):
    try:
        date, treatment, tube, zstack, side = simplify_names(name.strip())
    except:
        print(f'Error for {name}')
        assert False
        
    return {
        'date' : date,  # pd.Timestamp(tmp[1].split('-')[-1]),
        'treatment' : treatment,
        'tube' : tube,
        'zstack' : zstack,
        'side' : side if side != 'U' else '-'
    }

def exp_to_dates(n):
    '''
    exp 1: sett 2019; exp2: ott 2019; exp3: lug e sett 2020; exp4: dicembre 2020
    '''
    if n == 0: return ['4-12.09.19', '9-17.10.19', '29-30.07.2020', '2-3.09.2020', '11.12.2020', '18.12.2020']
    if n == 1: return ['4-12.09.19']
    if n == 2: return ['9-17.10.19']
    if n == 3: return ['29-30.07.2020', '2-3.09.2020']
    if n == 4: return ['11.12.2020', '18.12.2020']


def mean_density(df):
    means = np.array([df[(df.date==d) & (df.treatment=='CTRL')].density.mean() for d in df.date.unique() if 'CTRL' in df[df.date==d].treatment.unique()])
    idx = (np.abs(means - means.mean())).argmin()
    return means[idx]


def mcc(TP, TN, FP, FN, N=None):
    if N==None:
        N = TN + TP + FN + FP
    if N == 0: return 1
    S = (TP + FN)/N
    P = (TP + FP)/N
    return (TP/N - S*P) / np.sqrt(0.001 + P*S*(1-S)*(1-P))


def missed_wrong_cysts_dict(gt: np.array, pred: np.array, cutoff=0):
    cysts = []
    gt_contours, _ = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    gt_contours = tuple([c for c in gt_contours if c.size > 4 and cv2.contourArea(c)>cutoff])
    pred_contours = tuple([c for c in pred_contours if c.size > 4 and cv2.contourArea(c)>cutoff])

    gt_seps = tuple([csr_matrix(cv2.fillPoly(np.zeros_like(gt), pts=[c], color=(1))) for c in gt_contours])
    pred_seps = tuple([csr_matrix(cv2.fillPoly(np.zeros_like(gt), pts=[c], color=(1))) for c in pred_contours])
    
    for single_gt, c in zip(gt_seps, gt_contours):
        curr_detect_intersections = np.array([single_gt.multiply(sing_p).sum() for sing_p in pred_seps])
        # as a simplification, just keeping the one with the highest Intersection (not IoU)
        if (curr_detect_intersections > 0).any():
            best_p = np.argmax(curr_detect_intersections)
            cysts.append({'state': 'detected', 
                          'area_real': cv2.contourArea(c), 
                          'area_pred': cv2.contourArea(pred_contours[best_p]), 
                          'iou': curr_detect_intersections[best_p] / (single_gt + pred_seps[best_p]).astype(bool).sum(),
                         })
            for i, c_p in enumerate(pred_contours):
                if curr_detect_intersections[i] > 0 and i != best_p:
                    cysts.append({'state': 'overcounted',
                                  'area_real': cv2.contourArea(c), 
                                  'area_pred': cv2.contourArea(c_p), 
                                  'iou': None,
                                 })
        else: # missed cyst
            cysts.append({'state': 'missed',
                          'area_real': cv2.contourArea(c), 
                          'area_pred': None,
                          'iou': None,
                         })
    sparse_gt = csr_matrix(gt)
    for single_pred, c in zip(pred_seps, pred_contours):
        if not single_pred.multiply(sparse_gt).count_nonzero(): # wrong cyst
            cysts.append({'state': 'wrong',
                          'area_real': None, 
                          'area_pred': cv2.contourArea(c), 
                          'iou': None,
                         })
    return cysts, len(gt_contours), len(pred_contours)

## EVALUATOR

class Evaluator():
    def __init__(self, predP, maskP=None, imgP=None, annotP=None):
        self.imgP = imgP or Path.cwd() / 'artifacts/dataset:v0/images'
        self.maskP = maskP or Path.cwd() / 'artifacts/dataset:v0/masks'
        self.annotP = annotP or Path.cwd() / 'annotations_original/full_dataset'
        self.predP = predP
            
    def write_results(self):
        folder = self.predP.parent.parent
        datafile_im = folder / "images_table.csv"
        datafile_cy = folder / "cysts_table.csv"
        if datafile_im.exists():
            print(f"> Results tables in {folder.stem} already exists!")
            return

        IM_df = None
        CYST_df = None
        
        eval_name = folder.stem
        
        paths = sorted(self.predP.glob('*.png'))
    
        for pred in tqdm(paths, desc=str(folder)):
            name = pred.stem
            IM_s = pd.Series({"Analysis": eval_name}, name=name) 
            CYST_s = pd.Series({"Analysis": eval_name}) # Cyst row with {'state': state, AREA_real, AREA_pred, 'centers': [(x_r, y_r), (x_p, y_p)]}
            
            CYST_s["name"] = name
            for s, v in unpack_name(name).items():
                IM_s[s] = v
                CYST_s[s] = v

            # dict of cysts as {'state': state, 'areas': [AREA_real, AREA_pred]}
                
            assert (self.maskP / f'{name}.png').exists(), self.maskP / f'{name}.png'
            gt = load_mask(self.maskP / f'{name}.png')
            pred_img = load_mask(pred)
            
            cysts, IM_s['total_real'], IM_s['total_pred'] = missed_wrong_cysts_dict(gt, pred_img, cutoff=0)
            
            gt = gt.ravel()
            pred_img = pred_img.ravel()
            
            cf = confusion_matrix(gt, pred_img).ravel() if gt.any() else [0, 0, 0, 0]
            TN, FP, FN, TP = cf #if len(cf)==4 else [0, 0, 0, 0]
            IM_s['pxTP'] = int(TP)
            IM_s['pxFN'] = int(FN)
            IM_s['pxFP'] = int(FP)
            IM_s['pxTN'] = int(TN)
            
            IM_s['iou'] = float(TP / (TP + FN + FP + .001))
            IM_s['recall'] = float(TP / (TP + FN + .001))
            IM_s['precision'] = float(TP / (TP + FP + .001))
            IM_s['mcc'] = float(mcc(TP, TN, FP, FN))

            if IM_df is None: IM_df = pd.DataFrame(columns=IM_s.index)
            IM_df.loc[len(IM_df)] = IM_s

            if cysts: # skip if none of GT and Prediction have cysts
                if CYST_df is None: CYST_df = pd.DataFrame(columns=pd.concat([CYST_s, pd.Series(cysts[0])]).index)
                for c in cysts:
                    CYST_df.loc[len(CYST_df)] = pd.concat([CYST_s, pd.Series(c)])
            
        IM_df.to_csv(datafile_im)
        CYST_df.to_csv(datafile_cy)
        print(f'Results saved in "{datafile_im.parent}"')
        return

    def show(self, name, title=None):
        fig, ax = plt.subplots(1,3,figsize=(20,6))
        
        img = load_rgb(self.imgP / f"{name}.jpg")
        mask = load_mask(self.maskP / f"{name}.png")
        pred = load_mask(self.predP / f"{name}.png")
        
        if title: fig.suptitle(title)
        for a, im, tit in zip(ax, [img, mask, pred], ['img', 'mask', 'pred']):
            a.axis('off')
            a.set_title(tit)
            a.imshow(im)
        plt.show()