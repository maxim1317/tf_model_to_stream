import json
import requests

import utils.cognitive_face as CF
from termcolor import colored
import cv2
from utils.consts import *
from utils.misc import spc, ensure_dir


def checkIfTrained():
    '''
        checkIfTrained()
        ---
        Sends request FACE API to check person_group training status
    '''
    while (True):
        pass
        try:
            if CF.person_group.get_status(groupID)['status'] == 'succeeded':
                print(colored(' Checking if Trained...' + spc(21) + '[   ' , color='white') + colored('OK', color='green') + colored('   ]' , color='white'))
            else:
                print(colored(' Checking if Trained...' + spc(21) + '[ ' , color='white') + colored('FAILED', color='red') + colored(' ]' , color='white'))
            
            return 0
            
        except Exception as e:
            print(e)
            # time.sleep(wait_time)

def sendToDetection(frame):
    '''
        sendToDetection(frame)
        ---

    '''
    ensure_dir(pic_dir) # check if dir exists
    tmp_pic = pic_dir + 'pic_' + '0' +'.jpg'

    detected = []
    detected_faces = []
    
    cv2.imwrite(tmp_pic, frame) # Saving frame for a while

    try:
        # time.sleep(wait_time)
        detected = CF.face.detect(tmp_pic)
    except Exception as e:
        print(colored(e, color='grey'))
    print('#######################################################')
    print(colored(' Detection...' + spc(31) + '[   ' , color='white') + colored('OK', color='green') + colored('   ]' , color='white'))
        
    for detected_face in detected:
        detected_faces.append({'faceId' : detected_face['faceId'], 'faceRectangle': detected_face['faceRectangle']}) # Remembering face IDs and rectangles 
        print(colored(' Face detected:   ', color='white') + colored(detected_face['faceId'], color='yellow'))
        print('#######################################################') 
    return detected_faces

def sendToIdentification(detected_faces):
    
    identified_faces = []

    try:
        # time.sleep(wait_time)
        tmp_faces = CF.face.identify([d_f['faceId'] for d_f in detected_faces], groupID) # Trying to identify faces

        for identified_face in tmp_faces:
            if identified_face['candidates'][0]['confidence'] >= threshold:
                identified_faces.append(identified_face)

    except Exception as e:
        print(colored(e, color='grey'))

    print('#######################################################')
    if len(identified_faces) != 0:
        # identified_faces.sort()
        print(colored(' Identification...' + spc(26) + '[   ' , color='white') + colored('OK', color='green') + colored('   ]' , color='white'))
    else:
        print(colored(' Identification...' + spc(26) + '[ ' , color='white') + colored('FAILED', color='red') + colored(' ]' , color='white'))

    for identified_face in identified_faces:
        print(colored(' Face identified: ', color='white') + colored(identified_face['candidates'][0]['personId'], color='yellow')) 
    
    print('#######################################################')
    return identified_faces

def sendDataRequest(identified_face):
    person = 0

    while person == 0:
        try:
            # time.sleep(wait_time)
            person = CF.person.get(groupID, identified_face['candidates'][0]['personId']) # Getting person data
        except Exception as e:
            print(colored(e, color='grey'))
            person = 0

    print('#######################################################')
    print(colored(' Getting User Data...'+ spc(23) + '[   ' , color='white') + colored('OK', color='green') + colored('   ]' , color='white'))
    print('#######################################################')

    return person['userData']

def sendToFront(front_payload):

    while True:
        try:
            r = requests.post(front_url, headers=front_headers, data=json.dumps(front_payload))
            if r.status_code == requests.codes.ok:
                print('#######################################################')    
                print(colored(' Sending to Front...' + spc(24) + '[   ' , color='white') + colored('OK', color='green') + colored('   ]' , color='white'))
                print(front_payload)
                print(json.dumps(front_payload))
                print('#######################################################')
                return 0
            else:
                print('#######################################################')    
                print(colored(' Sending to Front...' + spc(24) + '[ ' , color='white') + colored('FAILED', color='red') + colored(' ]' , color='white'))
                print(front_payload)
                print(json.dumps(front_payload))
                print('#######################################################')

        except Exception as e:
            print('#######################################################')    
            print(colored(' Sending to Front...' + spc(24) + '[ ' , color='white') + colored('FAILED', color='red') + colored(' ]' , color='white'))
            print('#######################################################')
            print(colored(e, color='grey'))