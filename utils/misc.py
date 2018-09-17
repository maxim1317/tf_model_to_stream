

import os

# import sys

# import time
import wget

import cv2

# # import tensorflow as tf
# import numpy as np

from tqdm import tqdm
from termcolor import colored


from utils.consts import *

def spc(amount=0):
    '''
        spc(amount=0)
        ---
        Returns:
        <amount>_of_spaces for formating purposes     
    '''
    return ' ' * amount
    
def testDevice(source):
    '''
        testDevice(source)
        ---
        Returns:
        True/False if <source> is opened or failed to open accordingly
    '''
    cap = cv2.VideoCapture(source) 
    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)
        cap.release()
        return False
    return True

def ensure_dir(file_path):
    '''
        ensure_dir(file_path)
        ---
        Makes path to <file_path> if it doesn't exist

    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

readsize = 1024

def join(fromdir, tofile):
    output = open(tofile, 'wb')
    parts  = os.listdir(fromdir)
    parts.sort(  )

    bar = tqdm(total=len(parts))
    for filename in parts:
        filepath = os.path.join(fromdir, filename)
        fileobj  = open(filepath, 'rb')
        while 1:
            filebytes = fileobj.read(readsize)
            if not filebytes: break
            output.write(filebytes)
        fileobj.close(  )
        bar.update(1)
    bar.close()
    output.close(  )


def download_weights(model_dir):
    '''
    '''
    prefix = 'https://drive.google.com/uc?export=download&id='
    model_dir = model_dir
    tmp_dir = os.path.join(model_dir, 'tmp/')
    weights_file = os.path.join(model_dir, 'frozen_inference_graph.pb')

    # print('Looking for weights: ', weights_file)

    if not os.path.isfile(weights_file):
        print(colored(' Looking for Weights...' + spc(21) + '[ ' , color='white') + colored('FAILED', color='red') + colored(' ]' , color='white'))
        ensure_dir(weights_file)
        ensure_dir(tmp_dir)
        i = 1
        
        print(colored(' Loading Chunks...', 'yellow'))

        bar = tqdm(total=len(model_parts))
        for part in model_parts:
            wget.download(prefix+part, tmp_dir + 'part000'+str(i))
            i += 1
            bar.update(1)
        bar.close()

        print(colored(' Joining Chunks...', 'yellow'))
        join(tmp_dir, weights_file)

        print(colored(' Loading Weights...' + spc(25) + '[  ' , color='white') + colored('DONE', color='green') + colored('  ]' , color='white'))

    else:
        print(colored(' Looking for Weights...' + spc(21) + '[   ' , color='white') + colored('OK', color='green') + colored('   ]' , color='white'))
