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

drawable_classes = [
                    1, # Head
                    2, # Head
                    #3  # Person
                    ]

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

model_parts = [
   '1flRqp2GVU488yI0x-fYSlZnvGmShXCsY',
    '1RuZOb2FpN-5CF4WdKFi6bErtJdc4aeG1',
    '1r253hvKVbGgwu3mxqwnNXWx5rcRz6JsF',
    '1hFgEkjkFrYfdjTxn9tC4-hWV_dIWxZaG',
    '1AbqSGnlsp3pkLrUFTNstYHDdYN1bXQrJ',
    '1RnPZlDRfRenoKbzGvbno-kexbeOLp2Xw',
    '179qtG0-gp82iySwlEwGKJMwzYCSb24lk',
]

            

