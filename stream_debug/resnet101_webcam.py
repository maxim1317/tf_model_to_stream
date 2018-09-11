import cv2
import argparse
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--boxes_draw", default=0, type=int,
        help="way to draw bounding boxes(1 - tensorflow style, 0(default) is simple")
ap.add_argument('-vs', '--videostream', default='0', type=str,
                help='videofile path(or device number(default is 0))')
args = vars(ap.parse_args())

global vidcap
vidcap = cv2.VideoCapture(0)
vid_fps = vidcap.get(cv2.CAP_PROP_FPS)
print("Video fps: ", vid_fps)
print(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
W = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
H = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            
            try:

                        
                while vidcap.isOpened():
                            
                    _, frame = vidcap.read()
 
                    # out_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    _, buf = cv2.imencode(".jpg", frame)
                    self.wfile.write(b"--jpgboundary\r\n")
                    self.send_header('Content-type','image/jpeg')
                    self.send_header('Content-length',str(len(buf)))
                    self.end_headers()
                    self.wfile.write(bytearray(buf))
                    self.wfile.write(b'\r\n')


                    time.sleep(0.05)

            except KeyboardInterrupt:
                exit(0)
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
    vidcap.release()
    server.socket.close()
                
vidcap.release()
cv2.destroyAllWindows()
