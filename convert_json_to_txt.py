import json
import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

s = set()
def convert(folder,folder2,name):
    f = open(folder+'/'+name)
    os.system('touch '+folder2+'/'+name.split('.')[0]+'.txt')
    f2 = open(folder2+'/'+name.split('.')[0]+'.txt','w')
    img = cv2.imread(folder+'/'+name.split('.')[0]+'.jpg')
    #plt.imshow(img)
    #plt.show()
    data = json.load(f)
    #print(img.shape)
    #print(name)
    my_file = Path(folder+'/'+name.split('.')[0]+'.jpg')
    if my_file.is_file():
        #print(folder+'/combined/'+name.split('.')[0]+'.jpg')
        pass
    else:
        print(folder+'/'+name.split('.')[0]+'.jpg')
        return
    for i in data['shapes']:
        print(i)
        s.add(i['label'])
        if len(i['points'])==1:
            continue
        xmin = int(min(i['points'][0][0],i['points'][1][0]))
        xmax = int(max(i['points'][0][0],i['points'][1][0]))
        ymin = int(min(i['points'][0][1],i['points'][1][1]))
        ymax = int(max(i['points'][0][1],i['points'][1][1]))
        img=cv2.rectangle(img,(xmin,ymin),(xmax,ymax), [255,0,0], 3)
        #img = cv2.circle(img, (xmin,ymin), radius=0, color=(0, 0, 255), thickness=-1)
        img = cv2.line(img, (xmin,0), (xmin,3000), [0,255,0], 3)
        if i['label']=='label_tag'or i['label']=='cement_bag':
        # if i['label']=='person'or i['label']=='label_tag':

            #print('0 '+str(int(((xmin+xmax)/2)/img.shape[0]))+' '+str(int(((ymin+ymax)/2)/img.shape[1]))+' '+str(int((xmax-xmin)/img.shape[0]))+' '+str(int((ymax-ymin)/img.shape[1]))+'\n')
            f2.write('0 '+str((((xmin+xmax)/2)/img.shape[1]))+' '+str((((ymin+ymax)/2)/img.shape[0]))+' '+str(((xmax-xmin)/img.shape[1]))+' '+str(((ymax-ymin)/img.shape[0]))+'\n')
        '''
        else:
            f2.write('1 '+str((((xmin+xmax)/2)/img.shape[1]))+' '+str((((ymin+ymax)/2)/img.shape[0]))+' '+str(((xmax-xmin)/img.shape[1]))+' '+str(((ymax-ymin)/img.shape[0]))+'\n')
        '''
        #img = cv2.circle(img, (xmin,ymin), radius=0, color=(0, 0, 255), thickness=-1)
    #plt.imshow(img)
    #plt.show()
    # Closing file
    f.close()
    f2.close()
folder = './more_label_tag_annotations'
files = os.listdir(folder)
folder2=folder+'_yolo_format'
try:
    os.mkdir(folder2)
except:
    print('folder exists')
for j in files:
    print(j)
    if j.split('.')[-1]=='json':
        convert(folder,folder2,j)
print(s)
