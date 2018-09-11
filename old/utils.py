## IMPORTS

import argparse
import pickle
import socket
from io import BytesIO
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from object_detection.utils import ops as utils_ops


#### Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
            
## DETECTION
            

#### Counter
def draw_counter_on_frame(img_arr, txt_w, txt_cask, width=300, color=(89,89,89),
                          font_path = 'C:/Users/user/Documents/model_upd_win/win_files/arial.ttf'):
    imshape = (img_arr.shape[1], img_arr.shape[0])
    comb_size = (width + imshape[0], imshape[1])
    comb_img = Image.new('RGB', comb_size, color=color)
    comb_img.paste(Image.fromarray(img_arr))
    bottom_c = Image.open('C:/Users/user/Documents/model_upd_win/win_files/bottom_counter_proper.png')
    comb_img.paste(bottom_c,
                   box=(imshape[0],imshape[1] - 20 - bottom_c.size[1]))
    button_draw = ImageDraw.Draw(comb_img)
    button_draw.text((imshape[0] + 40, imshape[1]//2 - 200),
                     "Количество людей",
                     font=ImageFont.truetype(font_path, 25, ) )
    button_draw.rectangle([imshape[0]+1,100, comb_size[0]-1, 150],
                          fill = (255,255,255))
    button_draw.text((imshape[0] + 135, 100),
                     txt_w,
                     font=ImageFont.truetype(font_path, 45),
                     fill=(0,0,0) )
    button_draw.text((imshape[0] + 25, imshape[1]//2 - 50 ),
                 "Обнаружено без каски",
                 font=ImageFont.truetype(font_path, 25, ) )
    button_draw.rectangle([imshape[0]+1,imshape[1]//2 + 1, comb_size[0]-1, imshape[1]//2 + 51],
                          fill = (255,255,255))
    button_draw.text((imshape[0] + 135, imshape[1]//2 + 1),
                 txt_cask,
                 font=ImageFont.truetype(font_path, 45),
                 fill=(0,0,0) )
    return comb_img

def worker_count_from_dict(f_dict, thr = 0.75):
    num_det = f_dict['num_detections']

    prob_arr = f_dict['detection_scores'][:num_det]
    # prob_arr

    cl_arr = f_dict['detection_classes'][:num_det]
    # cl_arr

    prob_filt_arr = prob_arr[cl_arr==3]

    w_count =  prob_filt_arr[prob_filt_arr > thr].size
    return w_count
def wout_hh_count_from_dict(f_dict, thr = 0.75):
    num_det = f_dict['num_detections']

    prob_arr = f_dict['detection_scores'][:num_det]
    # prob_arr

    cl_arr = f_dict['detection_classes'][:num_det]
    # cl_arr

    prob_filt_arr = prob_arr[cl_arr==2]

    w_count =  prob_filt_arr[prob_filt_arr > thr].size
    return w_count

# def detect(vidcap, detection_graph):
     


                # if cv2.waitKey(25) & 0xFF == ord('q'):
                #     cv2.destroyAllWindows()
                #     out.release()
                #     vidcap.release()
                #     break
