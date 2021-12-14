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
    image_path = './put_image/nagyung.jpg'
    image = cv2.imread(image_path)
   # cv2.imshow("Image", image)


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
    #image = image[:, :, ::-1]  # bgr to rgb
    detector = FaceDetector() ## ok
    img, bboxes = detector.findFaces(image)
    show_img = img
    print(bboxes)

    #img = Image.fromarray(frame)
   # img = Image.fromarray(frame)
    cv2.imshow('img',image) # 480 640 3

    for i, b in enumerate(bboxes):
        print(i, b)
        x = b[0]
        y = b[1]
        w = b[2]
        h = b[3]

        person_img = image[y:y+h, x:x+w]
        cv2.imshow('crop',person_img)

        cv2.imwrite(os.path.join(info_path,'%s.jpg'%(name)),person_img)
        feature = np.squeeze(get_feature(person_img,model,trans,device))
        np.savetxt(os.path.join(info_path,'%s.txt'%(name)),feature)
    cv2.waitKey(0)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        return

class FaceDetector():
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        # min_detection_confidence - default=0.5

    def findFaces(self, img, draw = True):
        bboxs = []

      #  with self.mpFaceDetection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        self.results = self.faceDetection.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        if not self.results.detections:
            return img, []
        for detection in self.results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih)-50, \
                   int(bboxC.width * iw), int(bboxC.height * ih)+50
            bboxs.append(list(bbox))
            cv2.rectangle(img, bbox, (255,0,255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%',
                        (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)

        return img, bboxs
if __name__=='__main__':
    name = 'nagyung'
    save_person_information(name)
