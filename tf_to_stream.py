import argparse
import numpy as np
import cv2
import threading

import colorama

from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

from utils.cognitive_face import util as CF
from utils import *

import time


ap = argparse.ArgumentParser()
ap.add_argument('-vs', '--videostream', default='0', type=str,
                help='videofile path(or device number(default is 0))')
ap.add_argument('-hz', '--height_zone', default=0.0, type=float,
            help='fraction of height for search zone(default is 0.0))')
ap.add_argument('-wz', '--width_zone', default=0.5, type=float,
                help='fraction of width for search zone from middle point(default is 0.5))')

args = vars(ap.parse_args())


import tensorflow as tf

if int(tf.__version__.split('.')[1]) < 4:
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

## MODEL PREPARATION

#### Variables

#### Load a (frozen) Tensorflow model into memory.

download_weights(MODEL_NAME)

#### Video

full_vid1_path = args.get('videostream')
full_vid1_path = int(full_vid1_path) if full_vid1_path.isdigit() else full_vid1_path
# global vidcap
vidcap = cv2.VideoCapture(full_vid1_path)
vid_fps = vidcap.get(cv2.CAP_PROP_FPS)
vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
print("Video fps: ", vid_fps)
print(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
H = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#### Zone mults

h_mult = args.get('height_zone')
w_mult = args.get('width_zone')

colorama.init()

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
detection_sess = tf.Session(graph=detection_graph, config=config)

    






                
                
# vidcap.release()
# out.release()
# cv2.destroyAllWindows()

# inHelmet = False


def face_api(frame, inHelmet):
    detected_faces = []
    identified_faces = []

    front_payload = {
        'detectedPersons' : [],
        'inHelmet' : True
    }

    detected_faces = sendToDetection(frame)
    
    print()

    if len(detected_faces):
        # print('x')
        identified_faces = sendToIdentification(detected_faces) # Trying to identify faces
        print() 

        if len(identified_faces):
            for identified_face in identified_faces: 
                face_payload = {
                    'userData' : None,
                }            
                face_payload['userData'] = sendDataRequest(identified_face) # Getting user data
                front_payload['detectedPersons'].append(face_payload)
            print()

            front_payload['inHelmet'] = inHelmet

            sendToFront(front_payload) # Sending to Front

class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()

            inHelmet = False

            try:

                last_call_time = time.time()

                video_name = 'videofeed_proc_'+ str(int(time.time())) + '.mp4'
                out = cv2.VideoWriter(MODEL_NAME + video_name, fourcc, 6.0, (int(W),int(H)))
                ret,frame = vidcap.read()

                # label_color = { 1:( 44, 226, 166),
                #                 3:(255, 189, 137),
                #                 2:(114, 38 , 249)}
                cnt = 0                        
                while vidcap.isOpened():
                    ret, frame = vidcap.read()
                    out.write(frame)
                    # inHelmet = False

                    if not ret:
                        break

                    cur_time = time.time()

                    # else:
                    #     print('delta: ', last_call_time - cur_time)
                    # threading.Thread(target=model_tf, args=[frame, cnt]).start() 
                    # model_tf(frame, cnt)

                    # if cnt == 2:
                    #     timer = time.time()
                    print("Frame number: ", cnt)
                    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    im_w = image_np.shape[1]
                    im_h = image_np.shape[0]

                    z_xmin, z_xmax = int(im_w//2 - w_mult*im_w + 1), int(im_w//2 + w_mult*im_w)
                    z_ymin, z_ymax = int(im_h*h_mult), int(im_h)
                    print(z_xmin, z_xmax)
                    print(z_ymin, z_ymax)
                    z_h = z_ymax - z_ymin
                    z_w = z_xmax - z_xmin
                    zone_img = image_np.copy()[z_ymin:z_ymax, z_xmin:z_xmax]

                    if cur_time - last_call_time > cooling_time: 
                        threading.Thread(target=face_api, args=[zone_img, inHelmet]).start()
                        last_call_time = cur_time

                    inf_time = time.time()
                    image_np_expanded = np.expand_dims(zone_img, axis=0)
                    # Get handles to input and output tensors
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

                    # Run inference
                    boxes, scores, classes, num_detections = detection_sess.run(
                                                                                  [boxes, scores, classes, num_detections],
                                                                                  feed_dict={image_tensor: image_np_expanded}
                                                                               ) 
                    out_img = image_np.copy()
                    out_img = cv2.cvtColor(np.array(image_np.copy()), cv2.COLOR_RGB2BGR)
                    # im_w = image_np.shape[1]
                    # im_h = image_np.shape[0]
                    # zone = (
                    #     (int(im_w//2)- int(im_w*0.3), int(im_h//2)),
                    #     (int(im_w//2) + int(im_w*0.3), int(im_h - 1))
                    # )



                    cv2.rectangle(out_img, (z_xmin, z_ymin), (z_xmax - 1, z_ymax), (0, 240, 240), 1)

                    inHelmet=False
                    for ndet in range(int(num_detections[0])):
                        if scores[0][ndet] > 0.75:
                            ymin,xmin,ymax,xmax = boxes[0][ndet]
                            class_num = classes[0][ndet]
                            if class_num in drawable_classes:
                                # if class_num <= 3:
                                    cv2.rectangle( 
                                                   out_img,
                                                   (int(xmin*z_w) + z_xmin,int(ymin*z_h) + z_ymin),
                                                   (int(xmax*z_w) + z_xmin,int(ymax*z_h) + z_ymin),
                                                   label_color[class_num],
                                                   2
                                                )
                            if class_num == 1: 
                                inHelmet=True



                    zone_img = out_img.copy()[z_ymin:z_ymax, z_xmin:z_xmax]

                    out_img = cv2.blur(out_img, (40,40))
                    out_img[z_ymin:z_ymax, z_xmin:z_xmax] = zone_img
                    # cv2.rectangle( out_img,
                    #                (zone[0][0],zone[0][1]),
                    #                (zone[1][0],zone[1][1]),
                    #                (0,255,255),
                    #                1
                    #              )
                    print('inf time: ', time.time() - inf_time)

                    # imgRGB = out_img
                    r, buf = cv2.imencode(".jpg", out_img)
                    self.wfile.write(b"--jpgboundary\r\n")
                    self.send_header('Content-type','image/jpeg')
                    self.send_header('Access-Control-Allow-Origin','*')
                    self.send_header('Content-length',str(len(buf)))
                    self.end_headers()
                    self.wfile.write(bytearray(buf))
                    self.wfile.write(b'\r\n')

                    # out.write(out_img)
                    # time.sleep(0.05)


                    cnt += 1

            except Exception as e:
                print(e)
                # vidcap.release()
                out.release()
            return
        if self.path.endswith('.html') or self.path == "/":
            self.send_response(200)
            self.send_header('Content-type','text/html')
            self.send_header('Access-Control-Allow-Origin','*')
            self.end_headers()
            self.wfile.write('<html><head></head><body>')
            self.wfile.write('<img src="http://127.0.0.1:8086/cam.mjpg"/>')
            self.wfile.write('</body></html>')
            return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
	"""Handle requests in a separate thread."""

try:

    CF.Key.set(subscription_key)
    CF.BaseUrl.set(uri_base)


    
    print('\n#######################################################\n')
    checkIfTrained() # Checking if person_group is trained
    print('\n#######################################################\n')



    server = ThreadedHTTPServer(('localhost', 8086), CamHandler)
    print (" Server started")
    server.serve_forever()
except KeyboardInterrupt:
    # vidcap.release()
    server.socket.close()