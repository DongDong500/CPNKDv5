import os
import numpy as np
import traceback
from datetime import datetime

from train_mono import experiments as mono
from utils import MailSend
import utils

def train(opts) -> dict:
    ''' return(dict): {
                            "1" : {
                                "F1-0" : 0.1, "F1-1" : 0.9
                            },
                            "2" : {
                                "F1-0" : 0.1, "F1-1" : 0.9
                            },
                            "Overall F1[0] mean/std" : "0.2/0.01",
                            "Overall F1[1] mean/std" : "0.4/0.01"
                        }
    '''

    test_result = {}
    for exp_itr in range(0, opts.exp_itr):   
        print(f"{exp_itr+1}-th experiment")     
        run_id = exp_itr
        test_result[run_id] = mono(opts, run_id)
            
    f10 = 0
    f11 = 0
    s10 = 0
    s11 = 0
    N = 0
    for k in test_result.keys():
        N += 1
        f10 += test_result[k]['F1 [0]']
        f11 += test_result[k]['F1 [1]']
        s10 += test_result[k]['F1 [0]'] ** 2
        s11 += test_result[k]['F1 [1]'] ** 2
    f10 /= N
    f11 /= N
    s10 /= N
    s11 /= N
    # sample standard deviation
    s10 = np.sqrt((N/(N-1))*(s10 - f10**2))
    s11 = np.sqrt((N/(N-1))*(s11 - f11**2))

    test_result["Overall F1[0] mean/std"] = f"{f10:.8f}/{s10:.8f}"
    test_result["Overall F1[1] mean/std"] = f"{f11:.8f}/{s11:.8f}"
    
    return test_result

def exp(opts):
    params = save_argparser(opts, os.path.join(opts.default_prefix, opts.current_time))

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpus

    start_time = datetime.now()

    mlog = {}
    slog = {}

    mlog['Short Memo'] = opts.short_memo + '\n'
    slog['Short Memo'] = opts.short_memo + '\n'

    mlog['Experiments'] = train(opts)
    slog['Experiments'] = {
        'Overall F1[0] mean/std' : mlog['Experiments']['Overall F1[0] mean/std'],
        'Overall F1[1] mean/std' : mlog['Experiments']['Overall F1[1] mean/std']
    }
    slog['dir'] = opts.current_time

    params['Overall F1[0] mean/std'] = mlog['Experiments']['Overall F1[0] mean/std']
    params['Overall F1[1] mean/std'] = mlog['Experiments']['Overall F1[1] mean/std']

    time_elapsed = datetime.now() - start_time

    mlog['time elapsed'] = 'Time elapsed (h:m:s.ms) {}'.format(time_elapsed)
    slog['time elapsed'] = 'Time elapsed (h:m:s.ms) {}'.format(time_elapsed)
    params["time_elpased"] = str(time_elapsed)
    
    utils.save_dict_to_json(d=mlog, json_path=os.path.join(opts.default_prefix, opts.current_time, 'mlog.json'))
    utils.save_dict_to_json(d=params, json_path=os.path.join(opts.default_prefix, opts.current_time, 'summary.json'))

    # Transfer results by G-mail
    MailSend(subject = "Short report-%s" % "CPN segmentation exp results", 
                msg = slog,
                login_dir = opts.login_dir,
                ID = 'singkuserver',
                to_addr = ['sdimivy014@korea.ac.kr']).send()


if __name__ == '__main__':
    from args_mono import get_argparser, save_argparser

    total_time = datetime.now()
    ''' gaussian prior random crop size (360, 360) from (640, 640) 
        test with 1/2 scaled (360, 360) 
    '''
    try:
        is_error = False
        #size=(256, 256), normal_h=(21.08, 8.13), normal_w=(44.22, 12.01), block_size=5
        short_memo = ['10 New version of cpn (green dots on) (512, 512) base study n=20']
        for i in range(len(short_memo)):
            opts = get_argparser()
            opts.short_memo = short_memo[i]

            opts.dataset = 'cpn'
            
            opts.is_gaussian_crop = False

            opts.is_resize = False
            opts.is_resize_val = False
            opts.is_resize_test = False

            opts.is_crop = False
            opts.is_crop_val = False
            opts.is_crop_test = False

            opts.scale_factor = 0.5
            opts.scale_factor_val = 0.5
            opts.scale_factor_test = 0.5

            exp(opts)
        
    except KeyboardInterrupt:
        is_error = True
        print("Stop !!!")
    except Exception as e:
        is_error = True
        print("Error", e)
        print(traceback.format_exc())

    if is_error:
        os.rename(os.path.join(opts.default_prefix, opts.current_time), os.path.join(opts.default_prefix, opts.current_time + '_aborted'))
    
    total_time = datetime.now() - total_time
    print('Time elapsed (h:m:s.ms) {}'.format(total_time))