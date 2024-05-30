import argparse


class Args:
  def __init__(self):
    self.random_seed = 0
    self.workers = 4
    self.img_size = 224
    self.amp = True
    self.batch_size = 32
    self.max_epoch = 20
    self.lr = 1e-3
    self.decay = 0.001
    self.lr_factor = 0.5
    self.min_lr = 6e-6
    self.lr_schedule = "ReduceLROnPlateau"
    self.num_features=14
    self.embed_dim=24
    self.out_features = 37
    self.resume=False
    self.p = 0.3
    self.fct_path = 'models/fct.model'


def parse_args():
    parser = argparse.ArgumentParser(description='FCT for medical image')

    # msic
    # https://arxiv.org/pdf/2109.08203.pdf  3407 or other you like
    parser.add_argument('--random_seed',default=0,type=int,help='random seed,default 0')
    parser.add_argument("--workers", default=4, type=int, help="number of workers")
    parser.add_argument("--img_size",default=224,type=int)
    parser.add_argument("--amp",default=True,help="use automatic mix precision,defualt True")
    
    # train parameters
    parser.add_argument('--batch_size',default=32,type=int,help='batch size,default size 1')
    parser.add_argument('--max_epoch',default=120,type=int,help='the max number with epoch')
    parser.add_argument('--lr',default=1e-4,type=float,help='learning rate,when resume,set to 0 means using checkpoint lr,or a new lr')
    parser.add_argument('--decay',default=0.001,help='L2 norm')
    parser.add_argument('--lr_factor',default=0.5,help='dynamic learning rate factor,when loss not change,new lr = old_lr * lr_factor')
    parser.add_argument('--min_lr',default=6e-6,help='min dynamic learing rate')
    parser.add_argument('--lr_scheduler',default='ReduceLROnPlateau')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--new_param',action='store_true',help='use new param when resume from check point')

    parser.add_argument('--colab',action='store_true')
    # network parameters
    

    args = parser.parse_args()

    return args