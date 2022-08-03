import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class MailSend(object):

    def __init__(self, from_addr: list = ["singkuserver@gmail.com"], 
                to_addr: list = ["sdimivy014@korea.ac.kr"],
                subject: str = "Testing Mail system ... Do Not reply",
                msg: dict = {}, 
                attach: list = [],
                login_dir: str = None,
                ID = None, 
                ):
        """
        Args:
            from_addr: list of sender address
            to_addr: list of receiver address
            msg: Body message (type: dictionary)
            attach: list of attachment (images) directory
        """
        self.from_addr = from_addr
        self.to_addr = to_addr
        self.subject = subject
        self.message = msg
        self.attach = attach
        self.ID = ID
        self.login_dir = login_dir

        if os.path.exists(self.login_dir):
            with open(self.login_dir, "r") as f:
                self.users = json.load(f)
                self.PW = self.users[self.ID]
        else:
            raise RuntimeError("login file not found: ", self.login_dir)

    def send(self):
        """
        Args:

        Encryption Method
         TTL: smtplib.SMTP(smtp.gmail.com, 587)
         SSL: smtplib.SMTP_SSL(smtp.gmail.com, 465)

        """
        smtp = smtplib.SMTP('smtp.gmail.com', 587)
        smtp.starttls()
        smtp.login(self.ID, self.PW)

        msg = MIMEMultipart()
        msg['Subject'] = self.subject
        msg.attach(MIMEText('Auto mail transfer system ... \n\n', 'plain'))
        
        if isinstance(self.message, dict):
            for key, val in self.message.items():
                if isinstance(val, dict):
                    if isinstance(key, int):
                        msg.attach(MIMEText(str(key) + '-th results\n', 'plain'))
                    elif isinstance(key, str):
                        msg.attach(MIMEText(key + '\n', 'plain'))
                    else:
                        raise NotImplementedError   
                    for skey, sval in val.items():
                        msg.attach(MIMEText('\t' + skey + " : " + sval + '\n', 'plain'))
                elif isinstance(val, str):
                    msg.attach(MIMEText(val + '\n', 'plain'))
        elif isinstance(self.message, str):
            msg.attach(MIMEText(self.message, 'plain'))
        else:
            raise NotImplementedError

        smtp.sendmail(self.from_addr, self.to_addr, msg.as_string())

        smtp.quit()
        print("Sent e-mail to '{}'".format(self.to_addr[-1]))

    def append_msg(self, msg):
        if isinstance(msg, list):
            self.message.append(msg)
        elif isinstance(msg, dict):
            self.message.update(msg)
        else:
            self.message.append(str(msg))

    def append_from_addr(self, addr):
        self.from_addr.append(addr)

    def append_to_addr(self, addr):
        self.to_addr.append(addr)

    def reset(self):
        self.message = []

if __name__ == "__main__":

    
    sample_dict = {   1 : {"F1" : "[0.1, 0.9]",
                         "IoU" : "[0.5, 0.4]"},
                    "sub-a" : "sub"
                    }
    sample_dict_2 = {   2 : {"F1" : "[0.1, 0.9]",
                         "IoU" : "[0.5, 0.4]"},
                    "sub-b" : "sub"
                    }
    sample_str = '''
                    Fri Apr 29 15:13:19 2022
                    +-----------------------------------------------------------------------------+
                    | NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
                    |-------------------------------+----------------------+----------------------+
                    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
                    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
                    |                               |                      |               MIG M. |
                    |===============================+======================+======================|
                    |   0  GeForce RTX 3090    On   | 00000000:01:00.0 Off |                  N/A |
                    | 68%   65C    P2   291W / 350W |  20852MiB / 24268MiB |    100%      Default |
                    |                               |                      |                  N/A |
                    +-------------------------------+----------------------+----------------------+
                    |   1  GeForce RTX 3090    On   | 00000000:23:00.0 Off |                  N/A |
                    | 67%   65C    P2   296W / 350W |  20960MiB / 24268MiB |     97%      Default |
                    |                               |                      |                  N/A |
                    +-------------------------------+----------------------+----------------------+
                    |   2  GeForce RTX 3090    On   | 00000000:41:00.0 Off |                  N/A |
                    | 63%   62C    P2   333W / 350W |  20962MiB / 24268MiB |    100%      Default |
                    |                               |                      |                  N/A |
                    +-------------------------------+----------------------+----------------------+
                    |   3  GeForce RTX 3090    On   | 00000000:61:00.0 Off |                  N/A |
                    | 30%   28C    P8    19W / 350W |      1MiB / 24268MiB |      0%      Default |
                    |                               |                      |                  N/A |
                    +-------------------------------+----------------------+----------------------+
                    |   4  GeForce RTX 3090    On   | 00000000:81:00.0 Off |                  N/A |
                    | 61%   61C    P2   306W / 350W |  20962MiB / 24268MiB |    100%      Default |
                    |                               |                      |                  N/A |
                    +-------------------------------+----------------------+----------------------+
                    |   5  GeForce RTX 3090    On   | 00000000:A1:00.0 Off |                  N/A |
                    | 30%   30C    P8    19W / 350W |      1MiB / 24268MiB |      0%      Default |
                    |                               |                      |                  N/A |
                    +-------------------------------+----------------------+----------------------+
                    |   6  GeForce RTX 3090    On   | 00000000:C1:00.0 Off |                  N/A |
                    | 30%   27C    P8    22W / 350W |      1MiB / 24268MiB |      0%      Default |
                    |                               |                      |                  N/A |
                    +-------------------------------+----------------------+----------------------+
                    |   7  GeForce RTX 3090    On   | 00000000:E1:00.0 Off |                  N/A |
                    | 30%   31C    P8    23W / 350W |      1MiB / 24268MiB |      0%      Default |
                    |                               |                      |                  N/A |
                    +-------------------------------+----------------------+----------------------+

                    +-----------------------------------------------------------------------------+
                    | Processes:                                                                  |
                    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
                    |        ID   ID                                                   Usage      |
                    |=============================================================================|
                    |    0   N/A  N/A   4012154      C   python                          20845MiB |
                    |    1   N/A  N/A    576938      C   python                          20949MiB |
                    |    2   N/A  N/A    594236      C   python                          20951MiB |
                    |    4   N/A  N/A    721505      C   python                          20951MiB |
                    +-----------------------------------------------------------------------------+
    '''
    MailSend(from_addr=['singkuserver@gmail.com'],
                    to_addr=['sdimivy014@korea.ac.kr'],
                    msg=sample_dict,
                    login_dir='/data1/sdi/login.json',
                    ID='singkuserver').send()

    ms = MailSend(from_addr=['singkuserver@gmail.com'],
                    to_addr=['sdimivy014@korea.ac.kr'],
                    msg=sample_dict,
                    login_dir='/data1/sdi/login.json',
                    ID='singkuserver')
    ms.append_msg(sample_dict_2)
    ms.send()
