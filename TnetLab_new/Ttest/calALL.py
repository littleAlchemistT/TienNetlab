import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
from torchvision import transforms
from .evlmtrcs import Cal_evaluationMetrics


def testCal(net, data_loader, record_path = None, log_path = None, use_gpu = True
            , loss_func = None, per_img = False, if_curv = False, sep_plt = True, curv_elements = True
            , thrsh = 128, thrs_iter = range(256), pred2pic = False):
    with torch.no_grad():
        os.makedirs(record_path, exist_ok=True)
        if pred2pic:
            save_pic_path = os.path.join(record_path, 'pred_pics')
            os.makedirs(save_pic_path, exist_ok=True)

        if loss_func:
            epoch_losses = []

        net.cuda().eval() if use_gpu else net.eval()
        eva = Cal_evaluationMetrics(fixed_thr=thrsh, thrs_iter=thrs_iter, curves=if_curv, per_pic=per_img,
                                    save_curve_path=record_path)
        ###### iter loader
        for sample in tqdm(data_loader, ncols=70):
            img, target, img_name=sample['image'], sample['label'],sample['img_name'][0]
            target = np.array(transforms.ToPILImage()(target.squeeze().data.cpu()))
            if use_gpu:
                img=img.cuda()
            prediction=net(img).squeeze()
            
            if loss_func:
                epoch_losses.append(loss_func(prediction,target))
                
            prediction=torch.sigmoid(prediction).data.cpu()

            if img.shape!=target.shape:
                h, w = target.shape
                prediction = transforms.Resize((w,h))(transforms.ToPILImage()(prediction))
            prediction = np.array(prediction)
            assert target.shape == prediction.shape, f'tar:{target.shape}, pred:{prediction.shape}'
            assert target.dtype == prediction.dtype == 'uint8', f'tar:{target.dtype}, pred:{prediction.dtype}'

            ###### cal evaluation metrcs ######
            eva.process_cfMtrs(prediction, target, img_name=img_name.split('.')[0])
            print(img_name,'----')
            print('BER:', eva.BER(pred=prediction, label=target).values)
            print('mIoU:',eva.mIoU(pred=prediction, label=target).values)
            print('F1:',eva.F1(pred=prediction, label=target).values)
            print('Kappa:',eva.Kappa(pred=prediction, label=target).values)
            print('OA:',eva.OA(pred=prediction, label=target).values)

            if pred2pic:
                bi_pred = np.where(prediction >= thrsh, 255, 0)
                Image.fromarray(bi_pred.astype('uint8')).save(os.path.join(save_pic_path, img_name))
        
        rslt_BER = eva.BER()
        rslt_F1 = eva.F1()
        rslt_mIoU = eva.mIoU()
        rslt_Kappa = eva.Kappa()
        rslt_OA = eva.OA()
        
        if per_img:
            img_BER, BER=rslt_BER['per_pics'], rslt_BER['total']
            img_F1,F1 = rslt_F1['per_pics'],rslt_F1['total']
            img_mIoU, mIoU= rslt_mIoU['per_pics'],rslt_mIoU['total']
            img_Kappa, Kappa = rslt_Kappa['per_pics'],rslt_mIoU['total']
            img_OA, OA = rslt_OA['per_pics'],rslt_OA['total']

            perimg_metrics=pd.concat([img_BER,img_F1,img_mIoU,img_Kappa,img_OA],axis=1)
            total_metrcs=pd.concat([BER,F1,mIoU,Kappa,OA],axis=0)
            with open(file=log_path, mode='a') as f:
                f.write(f'\n{total_metrcs}\n{perimg_metrics}\n')
        else:
            total_metrcs=pd.concat([rslt_BER,rslt_F1,rslt_mIoU,rslt_Kappa,rslt_OA],axis=0)
            with open(file=log_path, mode='a') as f:
                f.write(f'\n{total_metrcs}\n')

        if if_curv:
            eva.preCal_curves()
            eva.PR(sep_plot=sep_plt, curv2pic=True)
            auc=eva.ROC(sep_plot=sep_plt,curv2pic=True)
            print(f'AUC:{auc}')
            if curv_elements:
                curve_elements = pd.concat([eva.p_df, eva.r_df, eva.fpr_df],axis=1)
                with open(file=log_path, mode='a') as f:
                    f.write(f'\nauc:{auc}\n{curve_elements}\n')
            else:
                with open(file=log_path, mode='a') as f:
                    f.write(f'\nauc:{auc}\n')
                    

        

