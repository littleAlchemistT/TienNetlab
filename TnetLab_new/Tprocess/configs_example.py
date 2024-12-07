import argparse
import os

exp_name=''
record_path=os.path.join('MyFile','rcd_'+exp_name)

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str,  default=exp_name, help='model_name')
parser.add_argument('--if_gpu',default=True)
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--train_root', type=str, default=r'autodl-tmp/AISD/train11836')
parser.add_argument('--val_root', type=str, default=r'autodl-tmp/AISD/Val51')
parser.add_argument('--test_root', type=str, default=r'autodl-tmp/AISD/Test51')
parser.add_argument('--resnext101_32_path',default = r'autodl-fs/resnext_101_32x4d.pth')

parser.add_argument('--record_path',default=record_path)
parser.add_argument('--log_file',default=record_path+'/log.txt')
parser.add_argument('--tb_path',default=r'tf-logs/'+exp_name)

parser.add_argument('--max_iterations', type=int,  default=10000)
parser.add_argument('--max_epoch', type=int,  default=10)
parser.add_argument('--scale', type=int,  default=256, help='batch size of 8 with resolution of 416*416 is exactly OK')
parser.add_argument('--img_norm', default={'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]})

parser.add_argument('--semi',default=True, help='labeled or semi')
parser.add_argument('--batch_size', type=int, default=6, help='per gpu')
parser.add_argument('--labeled_bs', type=int, default=4, help='per gpu')
parser.add_argument('--lss_type',default='Orig', help='Orig or bm_bce')
parser.add_argument('--edge_type',default='Orig',help='Orig or bm_bce')
parser.add_argument('--subit_type',default='Orig',help='Orig or big')

parser.add_argument('--w_decay',default=0.0001,help='0.0005 in paper, 0.0001 in code.')
parser.add_argument('--base_lr', type=float,  default=0.005, help='maximum epoch number to train')
parser.add_argument('--lr_decay', type=float,  default=0.9, help='learning rate decay')
parser.add_argument('--edge', type=float, default='10', help='edge learning weight')
parser.add_argument('--deterministic', type=int,  default=0, help='whether use it')
parser.add_argument('--seed', type=int,  default=1337, help='1337 random seed')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=7.0, help='consistency_rampup')
parser.add_argument('--subitizing', type=float,  default=1, help='subitizing loss weight')
parser.add_argument('--repeat', type=int,  default=3, help='repeat')

hyargs = parser.parse_args()



