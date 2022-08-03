from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import json
import os 

def cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def save_dict_to_json(d: dict, json_path: str):
    """Saves dict of floats in json file

    Args:
        d: dict
        json_path: (string) path to json file
    """
    if not os.path.exists(json_path):
        with open(json_path, 'w') as f:
            # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
            d = {k: v for k, v in d.items()}
            json.dump(d, f, indent=4)
    else:
        with open(json_path, 'r') as f:
            jdict = json.load(f)
        for key, val in d.items():
            jdict[key] = val
        with open(json_path, 'w') as f:
            json.dump(jdict, f, indent=4)

def txt_to_json(txt_path: str):
    """Convert 'summary.txt' to 'summary.json'
    """

    with open(txt_path, "r") as f:
        s = [x.split(" : ") for x in f.readlines()]
    
    jdict = {}
    for x in s:
        if len(x) == 2:
            jdict[x[0].strip()] = x[1].strip()
        else:
            jdict["Time elapsed"] = x

    save_dict_to_json(jdict, os.path.join(os.path.dirname(txt_path), 'summary.json'))

    return jdict

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


if __name__ == "__main__":

    """
    json_path = "/data1/sdi/CPNKD/utils/sample/summary.json"
    pram = Params(json_path=json_path)
    print(type(pram.Class_F1)) # list
    print(type(pram.Class_IoU)) # dict
    print(type(pram.separable_conv)) # bool
    print(type(pram.num_classes)) # int
    print(type(pram.weight_decay)) # float
    print(type(pram.ckpt)) # str
    print(pram.__dict__) # dict
    save_dict_to_json(pram.__dict__, "/data1/sdi/CPNKD/utils/sample/sample.json")

    pram = Params(json_path="/data1/sdi/CPNKD/utils/sample/sample.json")
    print(type(pram.Class_F1)) # list
    print(type(pram.Class_IoU)) # dict
    print(type(pram.separable_conv)) # bool
    print(type(pram.num_classes)) # int
    print(type(pram.weight_decay)) # float
    print(type(pram.ckpt)) # str
    print(pram.dict) # dict
    pram = Params(json_path="/data1/sdi/CPNKD/utils/sample/mlog.json")
    save_dict_to_json(pram.__dict__, "/data1/sdi/CPNKD/utils/sample/sample.json")
    """
    for x in os.listdir('/data1/sdi/CPNnetV1-result/'):
        for y in os.listdir(os.path.join('/data1/sdi/CPNnetV1-result/', x)):
            f_path = os.path.join('/data1/sdi/CPNnetV1-result/', x, y, 'summary.txt')
            txt_to_json(f_path)
    