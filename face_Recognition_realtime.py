import torch as t
from ArcFace.mobile_model import mobileFaceNet
from utils import cosin_metric, get_feature, draw_ch_zn
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image,ImageFont
import mediapipe as mp
import time


def verification():
    mpFaceDetection = mp.solutions.face_detection
    mpDraw = mp.solutions.drawing_utils
    faceDetection = mpFaceDetection.FaceDetection(0.75)
    # min_detection_confidence - default=0.5
    saved_model = './ArcFace/model/068.pth'
    name_list = os.listdir('./images')
    path_list = [os.path.join('./images',i,'%s.txt'%(i)) for i in name_list]
    total_features = np.empty((128,),np.float32)
    people_num = len(path_list)

    font = ImageFont.truetype('simhei.ttf',20,encoding='utf-8')

    if people_num>1:
        are = 'are'
        people = 'people'
    else:
        are = 'is'
        people = 'person'
    print('start retore users information, there %s %d %s information'%(are,people_num,people))
    for i in path_list:
        temp = np.loadtxt(i)
        total_features = np.vstack((total_features,temp))
    total_features = total_features[1:]

    # threshold = 0.30896
    threshold = 0.4
    model = mobileFaceNet()
    model.load_state_dict(t.load(saved_model)['backbone_net_list'])
    model.eval()
    use_cuda = t.cuda.is_available() and True
    device = t.device("cuda" if use_cuda else "cpu")

    # is_cuda_avilable
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
    pTime = 0

    while ret :
        #imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #results = faceDetection.process(imgRGB)
        img, bboxes = detector.findFaces(frame)
        recog = frame[:,:,::-1] #bgr to rgb
        #img = Image.fromarray(frame)
        # print(bbox)  # [[296.89171371 211.27569699 441.8924298  396.48678774   0.99999869]]
        print(bboxes)
        if len(bboxes) ==0:
            cv2.imshow('img', frame)
            # videoWriter.write(frame[:,:,::-1])
            cv2.waitKey(10)
            ret, frame = cap.read()
            continue

        for i, b in enumerate(bboxes):
            # print(i, b)
            x = b[0]
            y = b[1]
            w = b[2]
            h = b[3]
            person_img = recog[y:y+h, x:x+w].copy()
            if b[0]<0 or b[1]<0:
                continue
            feature = np.squeeze(get_feature(person_img, model, trans, device))
            cos_distance = cosin_metric(total_features, feature)
            index = np.argmax(cos_distance)

            if not cos_distance[index] > threshold:
                # 블러처리
                #print(cos_distance[index])
                #ret, frame = cap.read()
                #cv2.rectangle(frame, b, (255, 0, 255), 2)
                #cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255))
                continue
            else:
                #score = str(int(cos_distance[index]*100)) +"%"
                person = name_list[index]
                #cv2.rectangle(frame, b, (255, 0, 255), 2)
                roi = frame[y:y+h, x:x+w]
                roi = cv2.blur(roi, (20, 20))
                frame[y:y+h, x:x+w] = roi
                cv2.putText(frame,person,
                            (x, y-20), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (25, 25), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('img',frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            # videoWriter.release()
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
                cv2.rectangle(img, bbox, (255,0,255), 2)

        return img, bboxs

if __name__ =='__main__':

    verification()