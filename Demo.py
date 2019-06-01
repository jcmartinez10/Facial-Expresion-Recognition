#!/usr/bin/python3
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import urllib.request
import zipfile
from PIL import ImageFont, ImageDraw, Image 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



    


def process_landmarks(feature_vector):
    landmarks=[]
    xlist=[]
    ylist=[]
    for i in range(0,68):
        xlist.append(float(feature_vector.part(i).x))
        ylist.append(float(feature_vector.part(i).y))
    xmean = np.mean(xlist)
    ymean = np.mean(ylist)
    xcentral = [(x-xmean) for x in xlist]
    ycentral = [(y-ymean) for y in ylist]
    for j in range(0,68):
        for k in range(j+1,68):
            if j!=k:
                distx=(xcentral[j]-xcentral[k])
                disty=(ycentral[j]-ycentral[k])
                landmarks.append(distx)
                landmarks.append(disty)
    return landmarks

    

def detect_landmarks(img):
    import face_alignment
    import imutils
    import numpy as np
    import dlib
    import cv2
    
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    
    res = cv2.resize(img, dsize=(80, 80), interpolation=cv2.INTER_CUBIC)
    landmarks=np.empty([])
    rects = detector(res, 0)
    if (len(rects)) != 0:
        rect=rects[0]
        landmarks = predictor(res, rect)
        landmarks = process_landmarks(landmarks)
        landmarks=0.55*np.array(landmarks)
    else:
        rotated = imutils.rotate(res, 90)
        rects = detector(rotated, 0)
        if (len(rects)) != 0:
            rect=rects[0]
            landmarks = predictor(rotated, rect)
            landmarks=process_landmarks(landmarks)
            landmarks=0.55*np.array(landmarks)
        else:
            rotated = imutils.rotate(res, -90)
            rects = detector(rotated, 0)
            if (len(rects)) != 0:
                rect=rects[0]
                landmarks = predictor(rotated, rect)
                landmarks = process_landmarks(landmarks)
                landmarks=0.55*np.array(landmarks)
            else:
                preds = fa.get_landmarks(img)
                if preds is not None:
                    points = preds[0]
                    potints=np.array(points)
                    landmarks=[]
                    xl=[]
                    yl=[]
                    for i in range(0,68):
                        xl.append(points[i,0])
                        yl.append(points[i,1])
                    xmean = np.mean(xl)
                    ymean = np.mean(yl)
                    xcentral = [(x-xmean) for x in xl]
                    ycentral = [(y-ymean) for y in yl]
                    for j in range(0,68):
                        for k in range(j+1,68):
                            if j!=k:
                                distx=(xcentral[j]-xcentral[k])
                                disty=(ycentral[j]-ycentral[k])
                                landmarks.append(distx)
                                landmarks.append(disty) 
                    landmarks=np.array(landmarks)
                else:
                    preds = fa.get_landmarks(res)
                    if preds is not None:
                        points = preds[0]
                        potints=np.array(points)
                        landmarks=[]
                        xl=[]
                        yl=[]
                        for i in range(0,68):
                            xl.append(points[i,0])
                            yl.append(points[i,1])
                        xmean = np.mean(xl)
                        ymean = np.mean(yl)
                        xcentral = [(x-xmean) for x in xl]
                        ycentral = [(y-ymean) for y in yl]
                        for j in range(0,68):
                            for k in range(j+1,68):
                                if j!=k:
                                    distx=(xcentral[j]-xcentral[k])
                                    disty=(ycentral[j]-ycentral[k])
                                    landmarks.append(distx)
                                    landmarks.append(disty) 
                        landmarks=0.55*np.array(landmarks)
                        #print('Face align size: '+str(landmarks.size))
    return landmarks

def check_dependencies():
    if not os.path.isdir('./Extracted') or (not os.path.isfile('hybrid_model.pth')):
        print('Obtaining dependancies...')
        if not os.path.isfile('./dependencies.zip'):
            print('Starting download...')
            url = "https://www.dropbox.com/s/hyfqzbfid08p0jc/Dependencies.zip?dl=1" 
            u = urllib.request.urlopen(url)
            data = u.read()
            u.close()
            with open('dependencies.zip', "wb") as f :
                f.write(data)
            print('Download finished')
        zip_ref = zipfile.ZipFile('Dependencies.zip', 'r')
        zip_ref.extractall()
        zip_ref.close()
    ('Dependencies ready.')

def get_demo_image():

    x_demo, feature_demo = [], []

    path='./Demo'
    if os.listdir(path) == []:
        print('No images in demo folder, selecting image from data folder')
        path='./Extracted'
        
    filename=random.choice(os.listdir(path))
    
    if not os.path.isfile(filename+'_tensor.f') or not os.path.isfile(filename+'_landmarks.f'):
        
        img = Image.open(path+'/'+filename)
        img = img.resize((48, 48), Image.BICUBIC)
        landmarks=detect_landmarks(np.asarray(img))
        
        x_demo=img
        feature_demo=landmarks

        x_demo = torch.from_numpy(np.array(x_demo))
        feature_demo = torch.from_numpy(np.array(feature_demo))

        feature_demo = feature_demo.view(1,4556)
        torch.save(x_demo,filename+'_tensor.f')
        torch.save(feature_demo,filename+'_landmarks.f')
        
    else:
        feature_demo = torch.load(filename+'_landmarks.f')
        x_demo = torch.load(filename+'_tensor.f')
    return x_demo.to(device), feature_demo.to(device), filename,path


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg['VGG19'])
        self.joiner=nn.Linear(4556, 1024)
        self.classifier = nn.Linear(1536, 7)

    def forward(self, x, y):
        out = self.features(x)
        out2 = F.relu(self.joiner(y))
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = torch.cat((out, out2), 1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def run(model):
    if os.path.isfile('hybrid_model.pth'):
        model.load_state_dict(torch.load('hybrid_model.pth'))
    else:
        print('Warning: Pretrained model not found')
    x_test, feats_test,filename,path = get_demo_image()
    x_test=x_test.float()
    x_test=x_test.unsqueeze(0)
    x_test=x_test.permute(0, 3,1,2)
    feats_test=feats_test.float()
    outputs = model(x_test,feats_test)
    _, predicted = torch.max(outputs.data, 1)
    predicted=predicted.cpu()
    predicted=predicted.numpy()
    expression=predicted[0]
    labels = ['Neutral', 'Happy','Sad','Fearful', 'Angry', 'Surprised', 'Disgusted']
    label=labels[expression]
    

    img = Image.open(path+'/'+filename)
    resized=img.resize([256,256])
    draw=ImageDraw.Draw(resized)
    rest = label
    try:
        font = ImageFont.truetype("arial.ttf",30)
    except:
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf",30)
    w, h = draw.textsize(rest,font=font)
    draw.text(((256-w)/2,100), rest, font=font)
    im=plt.imshow(np.asarray(resized))
    plt.show()

if __name__ == '__main__':
    check_dependencies()
    model = VGG()
    model.to(device)
    run(model)
