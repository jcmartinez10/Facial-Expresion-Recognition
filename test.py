#!/usr/bin/python3
import os 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import zipfile
from skimage.measure import compare_ssim as ssim
from torch.autograd import Variable
from PIL import Image

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

def get_test_data():
    failures=[]
    
    x_test, y_test, feature_test = [], [], []

    if not os.path.isfile('./x_test.f'):
        print('Calculating features...')
        for filename in os.listdir('./Extracted'):
            label,subject = filename.split("_")
            subject=subject.replace('.jpg','')
            label=int(label)-1
            subject=int(subject)
            if subject>138:
                img = Image.open('./Extracted/'+filename)
                img = img.resize((48, 48), Image.BICUBIC)
                landmarks=detect_landmarks(np.asarray(img))
                if landmarks.size==4556:
                    img=np.asarray(img)/255
                    y_test.append(label)
                    x_test.append(img)
                    feature_test.append(landmarks)
                else:
                    failures.append(filename)
        workset=np.asarray(feature_test)
        average_features = np.mean(workset, axis=0)

        for j in failures:
            label,subject = filename.split("_")
            label=int(label)-1
            img = Image.open('./Extracted/'+filename)
            img = img.resize((48, 48), Image.BICUBIC)
            img=np.asarray(img)/255
            y_test.append(label)
            x_test.append(img)
            feature_test.append(average_features)


        x_test = torch.from_numpy(np.array(x_test))
        y_test = torch.from_numpy(np.array(y_test))
        feature_test = torch.from_numpy(np.array(feature_test))

        x_ttest = x_test.view(x_test.size(0), 3, 48,48)
        y_test = y_test.view(y_test.size(0), 1)
        feature_test = feature_test.view(feature_test.size(0),4556)
        torch.save(x_test,'x_test.f')
        torch.save(y_test,'y_test.f')
        torch.save(feature_test,'feature_test.f')
        
    else:
        feature_test = torch.load('feature_test.f')
        x_test = torch.load('x_test.f')
        y_test = torch.load('y_test.f')
        
    ('Test data ready.')
    return x_test.to(device), feature_test.to(device), y_test

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
    print('Getting test data...')
    x_test, feats_test, labels = get_test_data()
    x_test=x_test.float()
    x_test=x_test.permute(0, 3,1,2)
    feats_test=feats_test.float()
    results=np.empty([])
    model.eval()
    for i in range(0,x_test.shape[0], 100):
        _x_test = x_test[i:i+100]
        _feats_test = feats_test[i:i+100]
        with torch.no_grad():
            outputs = model(_x_test,_feats_test)
            _, predicted = torch.max(outputs.data, 1)
            predicted=predicted.cpu()
            predicted=predicted.numpy()
            if results.size==1:
                results=predicted
            else:
                results=np.append(results, predicted, axis=0)
  
    y_pred=results
    y_true=labels.numpy()
    from sklearn.metrics import accuracy_score
    aca=accuracy_score(y_true, y_pred)
    print('Accuracy: '+str(aca))

    from sklearn.metrics import confusion_matrix

    labels = ['Neutral', 'Happy','Sad','Fearful', 'Angry', 'Surprised', 'Disgusted']
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == '__main__':
    check_dependencies()
    model = VGG()
    model.to(device)
    run(model)
