from utils import *

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--boxes_draw", default=0, type=int,
        help="way to draw bounding boxes(1 - tensorflow style, 0(default) is simple")
ap.add_argument('-vs', '--videostream', default='0', type=str,
                help='videofile path(or device number(default is 0))')
args = vars(ap.parse_args())

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("C:/tensorflow/models/research/object_detection")

if int(tf.__version__.split('.')[1]) < 4:
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

## MODEL PREPARATION

#### Variables
MODEL_NAME = 'C:/Users/user/Documents/tf_model_to_stream/win_files/1399aug/'
PATH_TO_CKPT = MODEL_NAME + 'frozen_inference_graph.pb'
#### Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    
#### Video


full_vid1_path = args.get('videostream')
full_vid1_path = int(full_vid1_path) if full_vid1_path.isdigit() else full_vid1_path
global vidcap
vidcap = cv2.VideoCapture(full_vid1_path)
vid_fps = vidcap.get(cv2.CAP_PROP_FPS)
print("Video fps: ", vid_fps)
print(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
W = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
H = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# video_name = 'videofeed_proc_'+ str(int(time.time())) + '.mp4'

fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(MODEL_NAME + video_name, fourcc, 6.0, (int(W+300),int(H)))




class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()


            try:

                            
                video_name = 'videofeed_proc_'+ str(int(time.time())) + '.mp4'
                out = cv2.VideoWriter(MODEL_NAME + video_name, fourcc, 6.0, (int(W),int(H)))
                # img = detect(vidcap, detection_graph)

                ret,frame = vidcap.read()
                # timer = 0
                # label_color = { 1:(0,255,0),
                #                 2:(0,0,255),
                #                 3:(255,0,0) }
                label_color = { 1:( 44, 226, 166),
                                3:(255, 189, 137),
                                2:(114, 38 , 249)}
                cnt = 0

                with detection_graph.as_default():
                    with tf.Session() as sess:
                        
                        while vidcap.isOpened():
                            ret, frame = vidcap.read()
                            if not ret:
                                break
                            if (1) :
                                if cnt == 2:
                                    timer = time.time()
                                print("Frame number: ", cnt)
                                image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                inf_time = time.time()
                                # Get handles to input and output tensors
                                ops = tf.get_default_graph().get_operations()
                                all_tensor_names = {output.name for op in ops for output in op.outputs}
                                tensor_dict = {}
                                for key in [
                                    'num_detections', 'detection_boxes', 'detection_scores',
                                    'detection_classes', 'detection_masks'
                                ]:
                                    tensor_name = key + ':0'
                                    if tensor_name in all_tensor_names:
                                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                                            tensor_name)
                                
                                if 'detection_masks' in tensor_dict:
                                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                                        detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
                                    detection_masks_reframed = tf.cast(
                                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                                    # Follow the convention by adding back the batch dimension
                                    tensor_dict['detection_masks'] = tf.expand_dims(
                                        detection_masks_reframed, 0)
                                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                                # Run inference
                                output_dict = sess.run(tensor_dict,
                                                    feed_dict={image_tensor: np.expand_dims(image_np, 0)})

                                # all outputs are float32 numpy arrays, so convert types as appropriate
                                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                                output_dict['detection_classes'] = output_dict[
                                    'detection_classes'][0].astype(np.uint8)
                                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                                if 'detection_masks' in output_dict:
                                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

                                

                                wcnt = worker_count_from_dict(output_dict)
                                wout_hh_cnt = wout_hh_count_from_dict(output_dict)
                                # out_img = draw_counter_on_frame(image_np,
                                #                     txt_w = str(wcnt),
                                #                     txt_cask = str(wout_hh_cnt) )
                                out_img = image_np
                                out_img = cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR)

                                im_w = image_np.shape[1]
                                im_h = image_np.shape[0]
                                
                                print ("here 0")
                                for ndet in range(output_dict['num_detections']):

                                    if output_dict['detection_scores'][ndet] > 0.75:
                                        ymin,xmin,ymax,xmax = output_dict['detection_boxes'][ndet]
                                        class_num = output_dict['detection_classes'][ndet]
                                        if class_num <= 3:
                                            cv2.rectangle(out_img,
                                                        (int(xmin*im_w),int(ymin*im_h)),
                                                        (int(xmax*im_w),int(ymax*im_h)),
                                                        label_color[class_num],
                                                        2
                                                        )



                                # cv2.imshow('vid_even',out_img)
                                # data = pickle.dumps(frame)
                                # data = cv2.imencode('.jpg', out_img)[1].tostring()

                                print('time: ', time.time() - inf_time)

                                # imgRGB = out_img
                                r, buf = cv2.imencode(".jpg", out_img)
                                self.wfile.write(b"--jpgboundary\r\n")
                                self.send_header('Content-type','image/jpeg')
                                self.send_header('Access-Control-Allow-Origin','*')
                                self.send_header('Content-length',str(len(buf)))
                                self.end_headers()
                                self.wfile.write(bytearray(buf))
                                self.wfile.write(b'\r\n')
                                # rc,img = capture.read()
                                # if not rc:
                                    # continue
                                # imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                                # jpg = Image.fromarray(imgRGB)
                                # tmpFile = BytesIO()
                                # jpg.save(tmpFile,'JPEG')
                                # self.wfile.write(b"--jpgboundary")
                                # self.send_header(b'Content-type','image/jpeg')
                                # self.send_header(b'Content-length', str(len(tmpFile.getvalue())))
                                # self.end_headers()

                                out.write(out_img)
                                
                                print('here')

                                # jpg.save(self.wfile,'JPEG')
                                time.sleep(0.05)

                            cnt += 1


            except Exception as e:
                print(e)
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

global img_out

try:
    server = ThreadedHTTPServer(('localhost', 8086), CamHandler)
    print ("server started")
    server.serve_forever()
except KeyboardInterrupt:
    # vidcap.release()
    server.socket.close()
                
# timer = time.time() - timer
# print('\n -------------------------')
# print('Number of frames: ', cnt)
# print('Elapsed time: ', timer, ' s')
# print('Approximate fps: ', cnt/timer)
                
vidcap.release()
out.release()
cv2.destroyAllWindows()
