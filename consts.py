##################################################
################### API CONSTS ###################
##################################################

subscription_key = 'd83931a50c3c444da8deae372468f55f'
uri_base = 'https://northeurope.api.cognitive.microsoft.com/face/v1.0'

groupID = 'sibintek'

wait_time = 5
cooling_time = 1

##################################################
################## LOCAL CONSTS ##################
##################################################

FRAME_WIDTH = 720
FRAME_HEIGHT = 480

pics_needed = 6
threshold = 0.6
       
pic_dir = 'pics/frame/' # I save last frame here for cropping and stuff

##################################################
################## FRONT CONSTS ##################
##################################################

front_url = "http://localhost:9000/rest/userdata"

front_headers = {
    'Content-Type': 'application/json',
}

inHelmet = False

##################################################
############### TENSOR FLOW CONSTS ###############
##################################################

label_color = { 1:( 44, 226, 166),
                2:(114, 38 , 249),
                3:(255, 189, 137)}

##################################################
################# STORAGE CONSTS #################
##################################################

weights_file = 'win_files/1399aug/frozen_interference_graph.pb'
resnet_url   = 'https://yadi.sk/d/eOzlYwzh1iwWBA'

            

