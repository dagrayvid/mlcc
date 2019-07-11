from models import *
from utils import *

import os
import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import cv2
from PIL import Image

import argparse

parser = argparse.ArgumentParser(description='Run yolov3 frame by frame on video')

parser.add_argument('--cuda', dest='cuda', action='store_true', help='Run inference faster using CUDA')
parser.add_argument('--webcam-id', dest='webcam_id', default=0, help='webcam video device id number')
parser.add_argument('--input-video', help='Path to input video')
parser.add_argument('--output-video', help='Path to output video. Saving the results to a video will have a significant impact on performance due to encoding the video into a web friendly format.')
parser.add_argument('--no-display', dest='no_display', action='store_true', help='Dont show the video on the screen (for running in environments without a display)')

args=parser.parse_args()


# Global parameters
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.6
nms_thres=0.4


# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
if args.cuda:
    model.cuda()
model.eval()
classes = utils.load_classes(class_path)
if args.cuda:
    Tensor = torch.cuda.FloatTensor
else: 
    Tensor = torch.FloatTensor

cmap = cm.ScalarMappable('tab20b')
colors = [cmap.to_rgba(i, norm=False, bytes=True) for i in np.linspace(0, 1, 10)] #get 10 colors evenly spaced throughout the colormap
colors = [tuple(int(i) for i in tup[:3]) for tup in colors] # converting from RGBA uint8 to RGB int32
random.shuffle(colors) # randomize order of colors

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]



def cv_bounding_box(img, detections):

    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)

        # browse detections and draw bounding boxes
        color_index = 0
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]


            bb_color = colors[int(np.where(unique_labels == int(cls_pred))[0]) % len(colors)]
            color_index = (color_index + 1) % len(colors)

            # Draw bounding box, text, and background of text. It's a bit ugly...
            name=classes[int(cls_pred)]
            text_size=cv2.getTextSize(text=name, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
            cv2.rectangle(img,(x1,y1),(x1+box_w, y1+box_h), bb_color, 3)
            cv2.rectangle(img,(x1, y1), (x1+text_size[0][0], y1+1.5*text_size[0][1]), bb_color, cv2.FILLED)
            cv2.putText(img, name, (x1,y1 + text_size[0][1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=1,lineType=cv2.LINE_AA)

    return img

def main():
    # Start reading from video
    if args.input_video: 
        vc = cv2.VideoCapture(args.input_video)
    else:
        vc = cv2.VideoCapture(args.webcam_id)

    is_capturing, frame = vc.read()
    if args.output_video:
        out = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc('V','P','8','0'), 24, (frame.shape[1], frame.shape[0]))

    while(True) :
        is_capturing, frame = vc.read()
        if is_capturing:
            cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detect_image(Image.fromarray(cv_image))
            cv_boxed = cv_bounding_box(cv_image, detections)
            if not args.no_display:
                cv2.imshow('live detection', cv2.cvtColor(cv_boxed, cv2.COLOR_BGR2RGB))
            if args.output_video:
                out.write(cv2.cvtColor(cv_boxed, cv2.COLOR_RGB2BGR))

        
            if cv2.waitKey(1) == 27: 
                break  # esc to quit
        else:
            break

    if args.output_video:
        out.release()
    vc.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
