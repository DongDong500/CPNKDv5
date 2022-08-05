import os
import socket
import argparse
from datetime import datetime

import network
import datasets
import utils

HOSTNAME = {
    "server2" : 2,
    "server3" : 3,
    "server4" : 4,
    "server5" : 5
}
LOGIN = {
    "server2" : "/mnt/server5/sdi/login.json",
    "server3" : "/mnt/server5/sdi/login.json",
    "server4" : "/mnt/server5/sdi/login.json",
    "server5" : "/data1/sdi/login.json"
}
DEFAULT_PREFIX = {
    "server2" : "/mnt/server5/sdi",
    "server3" : "/mnt/server5/sdi",
    "server4" : "/mnt/server5/sdi",
    "server5" : "/data1/sdi"
}
DATA_DIR = {
    "server2" : "/mnt/server5/sdi/datasets",
    "server3" : "/mnt/server5/sdi/datasets",
    "server4" : "/mnt/server5/sdi/datasets",
    "server5" : "/data1/sdi/datasets"
}

def _get_argparser():
    parser = argparse.ArgumentParser()

    # prefix & path options
    parser.add_argument("--short_memo", type=str, default='short memo',
                        help="breif explanation of experiment (default: short memo")
    parser.add_argument("--cur_work_server", type=int, default=0,
                        help="current working server (default: 0)")
    parser.add_argument("--default_prefix", type=str, default='/',
                        help="path to results directory (default: /)")
    parser.add_argument("--current_time", type=str, default='current_time',
                        help="results images folder name (default: current_time)")
    parser.add_argument("--data_root", type=str, default='/',
                        help="path to Dataset root directory (default: /)")
    parser.add_argument("--login_dir", type=str, default='/',
                        help="E-mail log-in info json file (default: /)")
    parser.add_argument("--gpus", type=str, default="0,7",
                        help="gpus (default: 0,7)")

    # Tensorboard & store params options
    parser.add_argument("--Tlog_dir", type=str, default='/',
                        help="path to tensorboard log (default: /)")
    parser.add_argument("--save_model", action='store_false',
                        help="save best model param to \"./best-param\" (default: True)")
    parser.add_argument("--best_ckpt", type=str, default=None,
                        help="save best model param to \"./best-param\"")
    # Resume model from checkpoint
    parser.add_argument("--resume_ckpt", default='/', type=str,
                        help="resume from checkpoint (defalut: /)")
    parser.add_argument("--continue_training", action='store_true',
                        help="restore state from reserved params (defaults: false)")

    # Model options
    available_models = sorted(name for name in network.model.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.model.__dict__[name]) )
    parser.add_argument("--t_model", type=str, default='deeplabv3plus_resnet50', choices=available_models,
                        help='auxiliary model name (default: deeplabv3plus_resnet50)')
    parser.add_argument("--s_model", type=str, default='deeplabv3plus_resnet50', choices=available_models,
                        help='primary model name (default: deeplabv3plus_resnet50)')
    parser.add_argument("--t_model_params", type=str, default='/',
                        help="pretrained auxiliary network params (default: '/')")
    # DeeplabV3+ options
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="primary model: apply separable conv to decoder and aspp (default: False)")
    parser.add_argument("--output_stride", type=int, default=32, choices=[8, 16, 32, 64],
                        help="primary model: output stride (default: 32)")
    parser.add_argument("--t_separable_conv", action='store_true', default=False,
                        help="auxiliary model: apply separable conv to decoder and aspp (default: False)")
    parser.add_argument("--t_output_stride", type=int, default=32, choices=[8, 16, 32, 64],
                        help="auxiliary model: output stride (default: 32)")
    # Dataset options
    available_datasets = sorted( name for name in datasets.getdata.__dict__ if  callable(datasets.getdata.__dict__[name]) )
    parser.add_argument("--s_dataset", type=str, default="cpn", choices=available_datasets,
                        help='primary dataset (default: cpn)')                        
    parser.add_argument("--t_dataset", type=str, default="median", choices=available_datasets,
                        help='auxiliary dataset (default: median)')
    parser.add_argument("--s_dataset_ver", type=str, default="splits/v5/3",
                        help="version of primary dataset (default: splits/v5/3)")   
    parser.add_argument("--t_dataset_ver", type=str, default="splits",
                        help="version of auxiliary dataset (default: splits)")                     
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes (default: 2)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of workers (default: 8)")
    parser.add_argument("--is_rgb", action='store_false',
                        help="choose True: RGB, False: gray-scale (default: True)")
    parser.add_argument("--tvs", type=int, default=20,
                        help="The number of blocks where train set to be split (default: 20)")
    parser.add_argument("--pseudo_lbl_ratio", type=float, default=0.5,
                        help="ratio of pseudo label (default: 0.5)")
    # Transformation & Augmentation options
    parser.add_argument("--resize", default=(496, 468))
    parser.add_argument("--is_resize", action='store_false',
                        help="resize data (default: True)")
    parser.add_argument("--resize_val", default=(496, 468))
    parser.add_argument("--is_resize_val", action='store_false',
                        help="resize validate data (default: True)")
    parser.add_argument("--resize_test", default=(496, 468))
    parser.add_argument("--is_resize_test", action='store_false',
                        help="resize test data (default: True)")
    parser.add_argument("--crop_size", default=(512, 448))
    parser.add_argument("--crop_size_val", default=(512, 448))
    parser.add_argument("--crop_size_test", default=(512, 448))
    parser.add_argument("--gaussian_crop", action='store_true')
    parser.add_argument("--gaussian_crop_H", default=(0.0 ,0.1),
                        help='gaussian base random crop H (mean, std)')
    parser.add_argument("--gaussian_crop_W", default=(0.0 ,0.1),
                        help='gaussian base random crop W (mean, std)')
    parser.add_argument("--gaussian_crop_block_size", type=int, default=5)
    parser.add_argument("--scale_factor", type=float, default=5e-1)
    parser.add_argument("--scale_factor_val", type=float, default=5e-1)
    parser.add_argument("--scale_factor_test", type=float, default=5e-1)
    parser.add_argument("--c_std", type=float, default=0.0,
                        help="train sigma in gaussian noise channel (default: 0)")
    parser.add_argument("--c_mu", type=float, default=0.0,
                        help="train mean in gaussian noise channel (default: 0)")
    parser.add_argument("--std", type=float, default=0.0,
                        help="train sigma in gaussian perturbation (default: 0)")
    parser.add_argument("--mu", type=float, default=0.0,
                        help="train mean in gaussian perturbation (default: 0)")
    parser.add_argument("--std_val", type=float, default=0.0,
                        help="val sigma in gaussian perturbation (default: 0)")
    parser.add_argument("--mu_val", type=float, default=0.0,
                        help="val mean in gaussian perturbation (default: 0)")                
    parser.add_argument("--std_test", type=float, default=0.0,
                        help="test sigma in gaussian perturbation (default: 0)")
    parser.add_argument("--mu_test", type=float, default=0.0,
                        help="test mean in gaussian perturbation (default: 0)") 
    # Train options
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--total_itrs", type=int, default=2500,
                        help="epoch number (default: 2.5k)")
    parser.add_argument("--lr_policy", type=str, default='poly',
                        help="learning rate scheduler policy")
    parser.add_argument("--lr", type=float, default=1e-1,
                        help="learning rate (default: 1e-1)")
    parser.add_argument("--step_size", type=int, default=100, 
                        help="step size (default: 100)")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument("--optim", type=str, default='SGD',
                        help="optimizer (default: SGD)")
    parser.add_argument("--loss_type", type=str, default='kd_loss',
                        help="criterion (default: kd loss alpha=0 for ce+dl)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument("--exp_itr", type=int, default=2,
                        help='repeat N-identical experiments (default: 2)')    
    # Knowledge distillation
    parser.add_argument("--alpha", type=float, default=0,
                        help="alpha for KD loss (default: 0)")
    parser.add_argument("--T", type=float, default=3,
                        help="temperature in KD loss (default: 3)")
    # Validate options
    parser.add_argument("--val_interval", type=int, default=1,
                        help="epoch interval for eval (default: 1)")
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validate (default: 4)') 
    # Early stop options
    parser.add_argument("--patience", type=int, default=100,
                        help="Number of epochs with no improvement after which training will be stopped (default: 100)")
    parser.add_argument("--delta", type=float, default=0.001,
                        help="Minimum change in the monitored quantity to qualify as an improvement (default: 0.001)")
    # Test options
    parser.add_argument("--test_interval", type=int, default=1,
                        help="epoch interval for test (default: 1)")
    parser.add_argument("--test_batch_size", type=int, default=4,
                        help='batch size for test (default: 4)')
    parser.add_argument("--save_test_results", action='store_false',
                        help='save test results to \"./test\" (default: True)')
    parser.add_argument("--test_results_dir", type=str, default='/',
                        help="save segmentation results to (default: /)")
    
    # Run Demo
    parser.add_argument("--run_demo", action='store_true')

    return parser


def get_argparser(verbose=False):
    parser = _get_argparser().parse_args()

    hostname = socket.gethostname()
    s_folder = os.path.dirname( os.path.abspath(__file__) ).split('/')[-1] + '-result'
    current_time = datetime.now().strftime('%b%d_%H-%M-%S') + ('_demo' if parser.run_demo else '')
    _dir = os.path.join(DEFAULT_PREFIX[hostname], s_folder, current_time)

    if verbose:
        print(f'hostname: {hostname}\nfolder: {s_folder}\ncurrent time: {current_time}')

    if hostname not in HOSTNAME.keys():
        raise RuntimeError (f'hostname not found {hostname}')   
    
    parser.cur_work_server = HOSTNAME[hostname]
    parser.default_prefix = os.path.join(DEFAULT_PREFIX[hostname], s_folder)
    parser.current_time = current_time
    parser.data_root = DATA_DIR[hostname]
    parser.login_dir = LOGIN[hostname]
    parser.Tlog_dir = os.path.join(DEFAULT_PREFIX[hostname], s_folder, current_time, 'log')
    
    if not os.path.exists(parser.Tlog_dir):
        os.makedirs(parser.Tlog_dir)

    if parser.save_test_results:
        parser.test_results_dir = os.path.join(_dir, 'test')
        os.mkdir(parser.test_results_dir)

    if parser.save_model:
        parser.best_ckpt = os.path.join(_dir, 'best-param')
        os.mkdir(parser.best_ckpt)
    else:
        parser.best_ckpt = os.path.join(_dir, 'cache-param')
        os.mkdir(parser.best_ckpt)
        
    return parser
   
def save_argparser(parser, save_dir) -> dict:

    jsummary = {}
    for key, val in vars(parser).items():
        jsummary[key] = val

    utils.save_dict_to_json(jsummary, os.path.join(save_dir, 'summary.json'))

    return jsummary


if __name__ == "__main__":

    import utils

    print('basename:    ', os.path.basename(__file__)) # main.py
    print('dirname:     ', os.path.dirname(__file__)) # /data1/sdi/CPNKD
    print('abspath:     ', os.path.abspath(__file__)) # /data1/sdi/CPNKD/main.py
    print('abs dirname: ', os.path.dirname(os.path.abspath(__file__))) # /data1/sdi/CPNKD

    opts = get_argparser(verbose=True)
    jsummary = {}
    for key, val in vars(opts).items():
        jsummary[key] = val
    utils.save_dict_to_json(d=jsummary, json_path='/data1/sdi/CPNKDv4/utils/sample/opts_sample.json')

    pram = utils.Params('/data1/sdi/CPNKDv4/utils/sample/opts_sample.json')

    print(type(pram.separable_conv)) # bool
    print(type(pram.num_classes)) # int
    print(type(pram.weight_decay)) # float
    print(type(pram.best_ckpt)) # str

    pram.update(json_path='/data1/sdi/CPNKDv4/utils/sample/opts_sample.json')
    utils.save_dict_to_json(pram.__dict__, '/data1/sdi/CPNKDv4/utils/sample/out.json')