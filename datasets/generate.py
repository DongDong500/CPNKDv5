import os
import socket


DATA_DIR = {
    3 : "/mnt/server5/sdi/datasets",
    4 : "/mnt/server5/sdi/datasets",
    5 : "/data1/sdi/datasets"
}

def get_datadir():
    if socket.gethostname() == "server3":
        return DATA_DIR[3]
    elif socket.gethostname() == "server4":
        return DATA_DIR[4]
    elif socket.gethostname() == "server5":
        return DATA_DIR[5]
    else:
        raise NotImplementedError

if __name__ == "__main__":

    SIX = ['FH', 'FN', 'FN+1', 'FN+2', 'FN+3', 'FN+4']

    datadir = os.path.join(get_datadir(), 'CPN_all/Images')
    maskdir = os.path.join(get_datadir(), 'CPN_all/Masks')
    dstdir = os.path.join(get_datadir(), 'CPN_ver01/splits')

    imgs = [x for x in os.listdir(datadir)]
    masks = [x for x in os.listdir(datadir)]
    train_idx = val_idx = 0

    for x in os.listdir(datadir):
        if x.split('_')[0] in SIX:
            with open(os.path.join(get_datadir(), 'CPN_ver01/splits/train.txt'), "a") as f:
                f.write(x.split('.')[0]+'\n')
                train_idx += 1
        else:
            with open(os.path.join(get_datadir(), 'CPN_ver01/splits/val.txt'), "a") as f:
                f.write(x.split('.')[0]+'\n')
                val_idx += 1
    
    print("train: {}, val: {}".format(train_idx, val_idx))
        