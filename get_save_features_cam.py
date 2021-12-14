from ArcFace.mobile_model import mobileFaceNet
import cv2
import mediapipe as mp
import torch as t
from PIL import Image, ImageDraw
import numpy as np
from torchvision import transforms
import os
from utils_test import get_feature
import argparse


def save_person_information(name):
    saved_model = './ArcFace/model/068.pth'
    info_path = './images/'+name
    if not os.path.exists(info_path):
        os.makedirs(info_path)

    # threshold =  0.30896
    model = mobileFaceNet()
    model.load_state_dict(t.load(saved_model)['backbone_net_list'])
    model.eval()
    use_cuda = t.cuda.is_available() and True
    device = t.device("cuda" if use_cuda else "cpu")
    # is_cuda_avilableqq
    trans = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    model.to(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('failed open camara!!!')
    ret, frame = cap.read()
    detector = FaceDetector() ## ok
    while ret :
        img, bboxes = detector.findFaces(frame)
        show_img = img
        print(bboxes)
        frame = frame[:,:,::-1] # bgr to rgb
        #img = Image.fromarray(frame)
        img = Image.fromarray(frame)
        cv2.imshow('img',show_img) # 480 640 3
        for i, b in enumerate(bboxes):
            # print(i, b)
            x = b[0]
            y = b[1]
            w = b[2]
            h = b[3]
            if cv2.waitKey(1) & 0xFF == ord('c'):
                person_img = frame[y:y+h, x:x+w]
                cv2.imshow('crop',person_img[:,:,::-1])
                cv2.imwrite(os.path.join(info_path,'%s.jpg'%(name)),person_img[:,:,::-1])
                feature = np.squeeze(get_feature(person_img,model,trans,device))
                np.savetxt(os.path.join(info_path,'%s.txt'%(name)),feature)

            # cv2.waitKey(30)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()


class FaceDetector():
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        # min_detection_confidence - default=0.5

    def findFaces(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(img)
        #print(self.results)
        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih)-50, \
                       int(bboxC.width * iw), int(bboxC.height * ih)+50
                bboxs.append(list(bbox))
                #self.mp_drawing.draw_detection(img, detection)
                #print(bboxs)
                cv2.rectangle(img, bbox, (255,0,255), 2)
                cv2.putText(img, f'{int(detection.score[0]*100)}%',
                            (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)
            return img, bboxs
if __name__=='__main__':
    parse = argparse.ArgumentParser(description="get current user's face image")
    parse.add_argument('-n','--name',default=None,help='input current user\'s name')
    arg = parse.parse_args()
    name = 'nagyung'
    if name == None:
        raise ValueError('please input your name using \'python get_save_features.py --name your_name\'')
    save_person_information(name)