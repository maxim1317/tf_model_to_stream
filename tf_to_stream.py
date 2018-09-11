from utils_fused import *

def face_api(frame):
    detected_faces = []
    identified_faces = []
    front_payload = {
        'detectedPersons' : [],
        'inHelmet' : False
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
                    inHelmet = False

                    if not ret:
                        break

                    cur_time = time.time()

                    # else:
                    #     print('delta: ', last_call_time - cur_time)
                    # threading.Thread(target=model_tf, args=[frame, cnt]).start() 
                    # model_tf(frame, cnt)

                    if cnt == 2:
                        timer = time.time()
                    print("Frame number: ", cnt)
                    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    inf_time = time.time()
                    image_np_expanded = np.expand_dims(image_np, axis=0)
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
                    im_w = image_np.shape[1]
                    im_h = image_np.shape[0]
                    zone = (
                        (int(im_w//2)- int(im_w*0.3), int(im_h//2)),
                        (int(im_w//2) + int(im_w*0.3), int(im_h - 1))
                    )

                    for ndet in range(int(num_detections[0])):
                        if scores[0][ndet] > 0.75:
                            ymin,xmin,ymax,xmax = boxes[0][ndet]
                            class_num = classes[0][ndet]
                            if class_num in drawable_classes:
                                # if class_num <= 3:
                                    cv2.rectangle( 
                                                   out_img,
                                                   (int(xmin*im_w),int(ymin*im_h)),
                                                   (int(xmax*im_w),int(ymax*im_h)),
                                                   label_color[class_num],
                                                   2
                                                )
                            if class_num==1: inHelmet=True
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

                    if cur_time - last_call_time > cooling_time: 
                        threading.Thread(target=face_api, args=[frame]).start()
                        last_call_time = cur_time

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

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument('-vs', '--videostream', default='0', type=str,
                    help='videofile path(or device number(default is 0))')
    args = vars(ap.parse_args())

    colorama.init()

    download_weights()

    import tensorflow as tf

    if int(tf.__version__.split('.')[1]) < 4:
        raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

    ## MODEL PREPARATION

    #### Variables
    MODEL_NAME = os.abspath('model/')
    PATH_TO_CKPT = MODEL_NAME + 'frozen_inference_graph.pb'
    #### Load a (frozen) Tensorflow model into memory.

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
                    
                    
    vidcap.release()
    out.release()
    cv2.destroyAllWindows()
