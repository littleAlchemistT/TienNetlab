# based on numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc as AUC
import os


class Cal_evaluationMetrics:
    def __init__(self, fixed_thr=128, thrs_iter=range(256), curves=False, per_pic=False, save_curve_path=None):

        self.fixed_thr=fixed_thr
        self.thrs_iter = thrs_iter
        self.curves = curves
        self.per_pic = per_pic
        self.save_curve_path = save_curve_path

        self.fixed_mtrx = pd.Series([0, 0, 0, 0], ['TP', 'FP', 'TN', 'FN'])  ## for evl
        self.pics_mtrxs = pd.DataFrame() ## per pic. idx=cfmtrx, columns=pic names
        self.thrs_mtrxs=0  ## for curves. idx=cfmtrx, columns=thrs
        self.p_lst, self.r_lst, self.fpr_lst = None, None, None


    def core_cfMtrx(self, pred, label, thr):
        '''Receive ndarray'''
        pred = (pred >= thr).astype('float')  ## True/False → 0./1.
        label = (label >= 128).astype('float')

        TP_single = np.sum(pred * label)
        TN_single = np.sum((1 - pred) * (1 - label))
        FP_single = np.sum(pred * (1 - label))
        FN_single = np.sum((1 - pred) * label)
        assert TP_single >= 0 and TN_single >= 0 and FP_single >= 0 and FN_single >= 0, f'TP:{TP_single}, TN:{TN_single}, FP:{FP_single}, FN:{FN_single}'
        return pd.Series({'TP':TP_single,'TN': TN_single,'FP': FP_single,'FN': FN_single})
        

    def process_cfMtrs(self, pred, label,thr=None,thrs_iter=None,img_name=None):
        '''be iterated by dataloader'''

        thrs_iter = thrs_iter if thrs_iter else self.thrs_iter
        fixed_thr = thr if thr else self.fixed_thr

        single_cfsmtrx = self.core_cfMtrx(pred, label, fixed_thr)
        self.fixed_mtrx += single_cfsmtrx

        if self.per_pic:
            assert img_name
            self.pics_mtrxs[img_name] = single_cfsmtrx

        if self.curves:
            i_thrs_mtrxs={}
            for thr in thrs_iter:
                single_cfsmtrx = self.core_cfMtrx(pred, label, thr)
                i_thrs_mtrxs[f'thr{thr}'] = single_cfsmtrx
            self.thrs_mtrxs += pd.DataFrame(i_thrs_mtrxs)



    def preCal_curves(self,thrs_mtrxs=None):  # cal p、r、fpr
        '''after itering dataloader'''
        assert self.curves
        thrsMtrx = thrs_mtrxs if thrs_mtrxs else self.thrs_mtrxs

        ### r
        r_Serie = thrsMtrx.loc['TP', :] / (thrsMtrx.loc['TP', :] + thrsMtrx.loc['FN', :])
        r_Serie = r_Serie.replace('inf', 0)  # 'inf': gt all dark
        assert (0. <= r_Serie <= 1.).all(), f'Value should be between 0. and 1.'
        self.r_df = pd.DataFrame({'r':r_Serie})

        ### p
        p_Serie = thrsMtrx.loc['TP', :] / (thrsMtrx.loc['TP', :] + thrsMtrx.loc['FP', :])
        p_Serie = p_Serie.replace('inf', 0)  ## 'inf': pred all dark, thr=255
        assert (0. <= p_Serie <= 1.).all(), f'Value should be between 0. and 1.'
        self.p_df = pd.DataFrame({'p':p_Serie})

        ### fpr
        fpr_Serie = thrsMtrx.loc['FP'] / (thrsMtrx.loc['FP'] + thrsMtrx.loc['TN'])
        fpr_Serie = fpr_Serie.replace('inf', 0)  # 'inf'：gt all white
        assert (0. <= fpr_Serie <= 1.).all(), f'Value should be between 0. and 1.'
        self.fpr_df = pd.DataFrame({'fpr':fpr_Serie})



    def PR(self, sep_plot:bool, curv2pic:bool, label=None, color=None,r_lst=None,p_lst=None):
        assert self.curves
        if curv2pic:
            assert self.save_curve_path
        if sep_plot:
            plt.figure()
        plt.axis([-0.1, 1.1, -0.1, 1.1])
        plt.title('PR')
        plt.xlabel('R')
        plt.ylabel('P')

        if r_lst and p_lst:
            r_lst=r_lst
            p_lst=p_lst
        else:
            r_lst=self.r_df.values
            p_lst=self.p_df.values

        if label:
            plt.plot(r_lst, p_lst, label=label, color=color)
        else:
            plt.plot(r_lst, p_lst, color=color)
        plt.legend()

        if curv2pic:
            plt.savefig(os.path.join(self.save_curve_path, "PR.png"))


    def ROC(self, sep_plot:bool, curv2pic:bool, label=None, color=None,fpr_lst=None,r_lst=None):
        assert self.curves
        if curv2pic:
            assert self.save_curve_path

        if fpr_lst and r_lst:
            fpr_lst=fpr_lst
            r_lst=r_lst
        else:
            fpr_lst=self.fpr_df.values
            r_lst=self.r_df.values

        if sep_plot:
            plt.figure()
        plt.axis([-0.1, 1.1, -0.1, 1.1])
        plt.title('ROC')
        plt.xlabel('FPR')
        plt.ylabel('R')

        auc = round(AUC(fpr_lst, r_lst), 4)
        if label:
            text=str(label)+f'_AUC={auc}'
        else:
            text=f'AUC={auc}'

        plt.plot(fpr_lst, r_lst, label=text,color=color)
        plt.legend()

        if curv2pic:
            assert self.save_curve_path
            plt.savefig(os.path.join(self.save_curve_path, "ROC.png"))

        return auc



    def BER(self, pred=None, label=None, thr=None):
        '''smaller, better'''
        if pred is not None and label is not None:
            thrshld = thr if thr else self.fixed_thr

            cfmtrx = self.core_cfMtrx(pred, label, thrshld)
            sh_ER = (1.0 - cfmtrx.loc['TP'] / (cfmtrx.loc['TP'] + cfmtrx.loc['FN'])) * 100
            shlss_ER = (1.0 - cfmtrx.loc['TN'] / (cfmtrx.loc['TN'] + cfmtrx.loc['FP'])) * 100
            BER = 0.5 * (sh_ER + shlss_ER)
            total_BER=pd.Series([BER, sh_ER, shlss_ER], index=['BER', 'sh_ER', 'shlss_ER'])
            return total_BER

        else:
            sh_ER = (1.0 - self.fixed_mtrx.loc['TP'] / (self.fixed_mtrx.loc['TP'] + self.fixed_mtrx.loc['FN'])) * 100
            shlss_ER = (1.0 - self.fixed_mtrx.loc['TN'] / (self.fixed_mtrx.loc['TN'] + self.fixed_mtrx.loc['FP'])) * 100
            BER = 0.5 * (sh_ER + shlss_ER)
            total_BER = pd.Series([BER, sh_ER, shlss_ER], index=['BER', 'sh_ER', 'shlss_ER'])

            if self.per_pic:
                sh_ER_Serie = (1.0 - self.pics_mtrxs.loc['TP',:] / (self.pics_mtrxs.loc['TP',:] + self.pics_mtrxs.loc['FN',:])) * 100
                shlss_ER_Serie = (1.0 - self.pics_mtrxs.loc[ 'TN',:] / (self.pics_mtrxs.loc['TN',:] + self.pics_mtrxs.loc['FP', :])) * 100
                BER_Serie = 0.5 * (sh_ER_Serie + shlss_ER_Serie)

                return {'total':total_BER,'per_pics':pd.DataFrame({'BER':BER_Serie,'sh_ER':sh_ER_Serie, 'shlss':shlss_ER_Serie})}

            else:
                return total_BER



    def F1(self, pred=None, label=None, thr=None):
        '''bigger,better'''
        if pred is not None and label is not None:
            thrshld = thr if thr else self.fixed_thr
            cfmtrx = self.core_cfMtrx(pred, label, thrshld)

            r = cfmtrx.loc['TP'] / (cfmtrx.loc['TP'] + cfmtrx.loc['FN'])
            p = cfmtrx.loc['TP'] / (cfmtrx.loc['TP'] + cfmtrx.loc['FP'])
            return pd.Series({'F1':(2 * p * r) / (p+r)})

        else:
            r = self.fixed_mtrx.loc['TP'] / (self.fixed_mtrx.loc['TP'] + self.fixed_mtrx.loc['FN'])
            p = self.fixed_mtrx.loc['TP'] / (self.fixed_mtrx.loc['TP'] + self.fixed_mtrx.loc['FP'])
            f1 = pd.Series({'F1':(2 * p * r) / (p+r)})

            if self.per_pic:
                r_Serie = self.pics_mtrxs.loc['TP',:] / (self.pics_mtrxs.loc['TP',:] + self.pics_mtrxs.loc['FN',:])
                p_Serie = self.pics_mtrxs.loc['TP',:] / (self.pics_mtrxs.loc['TP',:] + self.pics_mtrxs.loc['FP',:])
                f1_Serie = (2 * p_Serie * r_Serie) / (p_Serie + r_Serie)
                return {'total':f1,'per_pics': pd.DataFrame({'F1':f1_Serie})}
            else:
                return f1


    def mIoU(self, pred=None, label=None, thr=None):
        '''bigger,better'''
        if pred is not None and label is not None:
            thrshld = thr if thr else self.fixed_thr
            cfmtrx = self.core_cfMtrx(pred, label, thrshld)
            iou_p = cfmtrx.loc['TP']/(cfmtrx.loc['TP']+cfmtrx.loc['FP']+cfmtrx.loc['FN'])
            iou_n = cfmtrx.loc['TN']/(cfmtrx.loc['TN']+cfmtrx.loc['FP']+cfmtrx.loc['FN'])
            return pd.Series({'mIoU':0.5 * (iou_n+iou_p)})

        else:
            assert self.fixed_thr
            iou_p = self.fixed_mtrx.loc['TP'] / (self.fixed_mtrx.loc['TP'] + self.fixed_mtrx.loc['FP'] + self.fixed_mtrx.loc['FN'])
            iou_n = self.fixed_mtrx.loc['TN'] / (self.fixed_mtrx.loc['TN'] + self.fixed_mtrx.loc['FP'] + self.fixed_mtrx.loc['FN'])
            bi_iou = pd.Series({'mIoU':0.5 * (iou_p + iou_n)})
            if self.per_pic:
                iou_p = self.pics_mtrxs.loc['TP',:] / (
                            self.pics_mtrxs.loc['TP',:] + self.pics_mtrxs.loc['FP',:] + self.pics_mtrxs.loc['FN',:])
                iou_n = self.fixed_mtrx.loc['TN',:] / (
                            self.pics_mtrxs.loc['TN',:] + self.pics_mtrxs.loc['FP',:] + self.pics_mtrxs.loc['FN',:])
                bi_iou_Serie = 0.5 * (iou_n+iou_p)

                return {'total': bi_iou, 'per_pics': pd.DataFrame({'mIoU':bi_iou_Serie})}
            else:
                return bi_iou


    def Kappa(self, pred=None, label=None, thr=None):
        '''bigger,better'''
        if pred is not None and label is not None:
            thrshld = thr if thr else self.fixed_thr
            cfmtrx = self.core_cfMtrx(pred, label, thrshld)

            gt_p=cfmtrx.loc['TP']+cfmtrx.loc['FN']
            gt_n=cfmtrx.loc['TN']+cfmtrx.loc['FP']
            pred_p=cfmtrx.loc['TP']+cfmtrx.loc['FP']
            pred_n=cfmtrx.loc['TN']+cfmtrx.loc['FN']

            Po = (cfmtrx.loc['TP'] + cfmtrx.loc['TN']) / (gt_n+gt_p)
            Pe = (gt_p * pred_p + gt_n * pred_n) / ((gt_n+gt_p) ** 2)
            return pd.Series({'Kappa':(Po - Pe) / (1 - Pe)})

        else:
            assert self.fixed_thr
            gt_p = self.fixed_mtrx.loc['TP'] + self.fixed_mtrx.loc['FN']
            gt_n = self.fixed_mtrx.loc['TN'] + self.fixed_mtrx.loc['FP']
            pred_p = self.fixed_mtrx.loc['TP'] + self.fixed_mtrx.loc['FP']
            pred_n = self.fixed_mtrx.loc['TN'] + self.fixed_mtrx.loc['FN']

            Po = (self.fixed_mtrx.loc['TP'] + self.fixed_mtrx.loc['TN']) / (gt_n + gt_p)
            Pe = (gt_p * pred_p + gt_n * pred_n) / ((gt_n + gt_p) ** 2)
            kappa = pd.Series({'Kappa':(Po - Pe) / (1 - Pe)})

            if self.per_pic:
                gt_p = self.pics_mtrxs.loc['TP',:] + self.pics_mtrxs.loc['FN',:]
                gt_n = self.pics_mtrxs.loc['TN',:] + self.pics_mtrxs.loc['FP',:]
                pred_p = self.pics_mtrxs.loc['TP',:] + self.pics_mtrxs.loc['FP',:]
                pred_n = self.pics_mtrxs.loc['TN',:] + self.pics_mtrxs.loc['FN',:]

                Po_Serie = (self.pics_mtrxs.loc['TP',:] + self.pics_mtrxs.loc['TN',:]) / (gt_n + gt_p)
                Pe_Serie = (gt_p * pred_p + gt_n * pred_n) / ((gt_n + gt_p) ** 2)
                kappa_Serie=(Po_Serie - Pe_Serie) / (1 - Pe_Serie)
                return {'total':kappa,'per_pics':pd.DataFrame({'Kappa':kappa_Serie})}
            else:
                return kappa



    def OA(self, pred=None, label=None, thr=None):
        '''bigger,better'''
        if pred is not None and label is not None:
            thrshld = thr if thr else self.fixed_thr
            cfmtrx = self.core_cfMtrx(pred, label, thrshld)
            return pd.Series({'OA':(cfmtrx.loc['TP'] + cfmtrx.loc['TN']) / cfmtrx.sum()})

        else:
            assert self.fixed_thr
            oa = pd.Series({'OA':(self.fixed_mtrx.loc['TP']+self.fixed_mtrx.loc['TN'])/self.fixed_mtrx.sum()})
            if self.per_pic:
                oa_Serie=(self.pics_mtrxs.loc['TP',:]+self.pics_mtrxs.loc['TN',:])/self.pics_mtrxs.sum(axis=0)
                return {'total':oa, 'per_pics': pd.DataFrame({'OA':oa_Serie})}
            else:
                return oa






