#!/usr/bin/env python
from darkflow.net.build import TFNet
import cv2
import os
import scp
import random
from subprocess import Popen
import time
from PIL import Image,ImageDraw
import numpy as np
import paho.mqtt.client as mqtt
import argparse
import ai_onvifconfig
import socket
#In case Raspberry Pi camera is used instead of RTSP stream
import picamera
usepicamera = False

#RTSP captured frame frame_filename; preferably on RAM drive
#diskutil erasevolume HFS+ 'RAM Disk' `hdiutil attach -nomount ram://20480`
#frame_filename = '/Volumes/RAM Disk/frame'+str(random.randint(1,99999))+'.jpeg'

#Raspberry Pi
#sudo mkdir /tmp/ramdisk; sudo chmod 777 /tmp/ramdisk
#sudo mount -t tmpfs -o size=16M tmpfs /tmp/ramdisk/

frame_filename = '/tmp/ramdisk/frame'+str(random.randint(1,99999))+'.jpeg'

#threshold parameter is below, keeping it too low will result in recognition errors
options = {"model": "/darkflow/cfg/tiny-yolo-voc.cfg", "load": "/darkflow/tiny-yolo-voc.weights",  "threshold": 0.55}

ptz = ai_onvifconfig.ptzcam()

parser=argparse.ArgumentParser()
parser.add_argument(
  "--watch",  # name on the parser - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=str,
  default=['person', 'cat', 'dog', 'bird'],  # default if nothing is provided
)

parser.add_argument(
  "--stream",  # name on the parser - drop the `--` for positional/required parameters
  type=str,
  default='rtsp://admin:password@192.168.0.61:10554/udp/av0_0', # default if nothing is provided
)

parser.add_argument(
  "--broker",  # name on the parser - drop the `--` for positional/required parameters
  type=str,
  default= '', # default if nothing is provided
)

parser.add_argument(
  "--topic",  # name on the parser - drop the `--` for positional/required parameters
  type=str,
  default= '', # default if nothing is provided
)

parser.add_argument(
  "--showimage",  # name on the parser - drop the `--` for positional/required parameters
  type=str,
  default='no', # default if nothing is provided
)

# parse the command line
args = parser.parse_args()
watch_list=args.watch
rtsp_stream=args.stream
broker_address=args.broker
mqtt_topic=args.topic
showimageflag=args.showimage

print("Watching for: %r" % watch_list)
print("Stream: %r" % rtsp_stream)
print("MQTT broker: %r" % broker_address)
print("MQTT topic: %r" % mqtt_topic)
print("Show image: %r" % showimageflag)

'''
if usepicamera:
    camera = picamera.PiCamera()
    camera.resolution = (640, 480)
else:
    #start a ffmpeg process that captures one frame every 2 seconds
    p = Popen(['ffmpeg', '-loglevel', 'panic', '-rtsp_transport', 'udp', '-i', rtsp_stream, '-f' ,'image2' ,'-s', '640x480', '-pix_fmt', 'yuvj420p', '-r', '1/2' ,'-updatefirst', '1', frame_filename])
'''

if broker_address!='':
    client = mqtt.Client("cameraclient_"+str(random.randint(1,99999999))) #create new instance
    client.connect(broker_address)

tfnet = TFNet(options)

while True:
    temp = 0
    os.system("ffmpeg -rtsp_transport -udp_multicast -i "+rtsp_stream+" -f image2 -s 1280x720 -pix_fmt yuvj420p "+frame_filename)
    try:
        if usepicamera:
            camera.capture( frame_filename )
        curr_img = Image.open( frame_filename )
        curr_img_cv2 = cv2.cvtColor(np.array(curr_img), cv2.COLOR_RGB2BGR) #is the frame good and can be opened?
        #curr_img_cv2 = cv2.resize(curr_img_cv2, (640, 360)) 
        os.remove(frame_filename) #delete frame once it is processed, so we don't reprocess the same frame over
    except: # ..frame not ready, just snooze for a bit
        time.sleep(1)
        continue

    result = tfnet.return_predict(curr_img_cv2)
    print(result)

    if broker_address!='':
        client.publish(mqtt_topic, "".join([str(x) for x in result]) ) #publish

    saveflag=True
    namestr=''
    criminal_YN = ''
    draw = ImageDraw.Draw(curr_img)
    for det_object in result:
        middle_x = 0
        middle_y = 0
        pan = 0
        tilt = 0
        timeout = 0
        area = (det_object['topleft']['x'], det_object['topleft']['y'], det_object['bottomright']['x'], det_object['bottomright']['y'])
        namestr+='_'+str(det_object['label'])
        if det_object['label'] == 'person':
            temp+=1
            # person img crop & save
            cropped_img = curr_img.crop(area)
            filename_temp = str(int(time.time()))+'_'+str(temp)
            crop_filename = '/home/pi/Pictures/'+filename_temp+'.jpg'
            cropped_img.save(crop_filename)
            # file transmission to Ubuntu
            os.system("sshpass -ppassword scp -o StrictHostKeyChecking=no "+crop_filename+" banana@192.168.0.55:/home/tmpscp/temp")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('192.168.0.55', 10999))
            sock.send(filename_temp.encode())
            print("img transmit OK..")
            criminal_YN = (sock.recv(65535)).decode('utf-8')
            # fine the middle point & PTZ control
            if criminal_YN == 'yes':
                print("criminal : yes")
                middle_x = (det_object['topleft']['x'] + det_object['bottomright']['x']) / 2
                middle_y = (det_object['topleft']['y'] + det_object['bottomright']['y']) / 2
                if middle_x >864:
                    pan = 1
                    timeout = 3
                elif middle_x > 704:
                    pan = 1
                    timeout = 1
                elif middle_x < 416:
                    pan = -1
                    timeout = 3
                elif middle_x < 576:
                    pan = -1
                    timeout = 1
                '''
                if middle_y > 480:
                    tilt = -1
                elif middle_y < 240:
                    tilt = 1
                '''
                ptz.move_PT(pan, tilt, timeout)
            elif criminal_YN == 'no':
                print("criminal : no")
            else:
                break
    for det_object in result:
        #if any(det_object['label'] in s for s in watch_list):
            draw.rectangle([det_object['topleft']['x'], det_object['topleft']['y'], det_object['bottomright']['x'], det_object['bottomright']['y']], outline=(255, 255, 0))
            draw.text([det_object['topleft']['x'], det_object['topleft']['y'] - 13], det_object['label']+' - ' + str(  "{0:.0f}%".format(det_object['confidence'] * 100) ) , fill=(255, 255, 0))
    if saveflag == True:
        curr_img.save('/home/pi/Pictures/'+str(int(time.time()))+namestr+'.jpg')
        saveflag=False
    if criminal_YN == 'exception':
        print("Exception occur !") # Unable to load img, find a face, align img
        continue
    if showimageflag!='no':
        curr_img_cv2=cv2.cvtColor(np.array(curr_img), cv2.COLOR_RGB2BGR)
        #curr_img_cv2 = cv2.resize(curr_img_cv2, (640, 360)) 
        cv2.imshow("Security Feed", curr_img_cv2)
        if cv2.waitKey(50) & 0xFF == ord('q'): # wait for image render
            break
            continue
#    time.sleep(1)
p.terminate()
cv2.destroyAllWindows()
