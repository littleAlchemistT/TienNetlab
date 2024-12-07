''' Cal the evaluations for testing or validating.
E.G.: <type the below in your own test py file>
    1. from TnetLab import Ttest

    2. kwards_dict={....}
       Ttest.set_kwards(kwards_dict)
    3. Ttest.testAll()

'''

from .utils import testDataLoader
from .calALL import testCal


test_kwards = {'if_gpu': True
    , 'net': None         # trained model.
    , 'Dataloader': None   # your own test dataloader. Default: this library's dataloader.
    , 'loss_func': None
    , 'record_path': None # record path for evaluating results
    , 'log_file':None
               
    ###### If use built-in dataloader, set the following parameters↓.
    , 'dataPath': None      # val or test data path
    , 'resize':None   # resize data
    , 'data_norm':None  # torchvision.transforms.Normalize
    , 'imgFolder_name': 'shadow'
    , 'targetFolder_name': 'mask'


    ###### cal eva mtrcs ######
    , 'thrshd':128
               
    , 'thr_iter':range(256)
    , 'if_curv':False   # whether cal curves
    , 'sep_plt':True   # whether set a new plt window.
    , 'curv2pic':True
    , 'curv_elements':False  # log curve elements
    , 'per_img':False   # cal per img metrix
    , 'pred2pic':False
               }

def set_kwards(kwards_dict):
    for k in kwards_dict:
        test_kwards[k]=kwards_dict[k]


def _get_dataloader():
    if test_kwards['Dataloader']:
        print('Input dataloader')
        return test_kwards['Dataloader']
    
    else:
        print('built-in dataloader')
        assert test_kwards['dataPath'] , "Please set datapath."

        return testDataLoader(test_kwards['dataPath']
                               , subfile=[test_kwards['imgFolder_name'],test_kwards['targetFolder_name']]
                               , resize=test_kwards['resize'], img_norm=test_kwards['data_norm']
                          )

def testAll():
    testCal(net=test_kwards['net'], data_loader=_get_dataloader(), record_path=test_kwards['record_path']
         ,log_path=test_kwards['log_file'], use_gpu=test_kwards['if_gpu']
            , loss_func=test_kwards['loss_func'],per_img=test_kwards['per_img']
         , if_curv=test_kwards['if_curv'],sep_plt=test_kwards['sep_plt'], curv_elements=test_kwards['curv_elements']
         , thrsh=test_kwards['thrshd'], thrs_iter=test_kwards['thr_iter']
         ,pred2pic=test_kwards['pred2pic']
         )



    

            




