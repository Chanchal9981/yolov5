import os 
import random
import cv2
import torch

# img_dir = "/home/frinks1/Downloads/DP/Heidelberg/ratio_test/negative_images/"

img_dir = "/home/frinks1/Downloads/DP/Heidelberg/bag_counting/jsw/bags/data/"
OUTOUT_DIRR  = "./auto_annotations"
os.makedirs(OUTOUT_DIRR, exist_ok= True)

DETECT_ONLY_AFTER_CX =  0 ## 245
DETECT_ONLY_AFTER_CY = 0
SCORE_THRESHOLD_BAG, IOU_THRESHOLD_BAG =  0.3, 0.3 # 0.15,0.3 #### FOR 0.5 some detections are missing
SCORE_THRESHOLD_TAG, IOU_THRESHOLD_TAG = 0.5, 0.3 # 0.7, 0.3 iou low value gives good result

ISRATIO = False

# BAG_MODEL_WEIGHT = "/home/frinks1/Downloads/DP/Heidelberg/minimum_data_test/bag_counting/yolov5l_training_results/training_backup_400_data_640ims_beste_cc_10p/weights/epoch150.pt"
# BAG_MODEL_WEIGHT = '/home/frinks1/Downloads/DP/Heidelberg/bag_counting/yolov5l_training_results/training_backup_503b_data_640ims_beste_cc_22/weights/best.pt'

BAG_MODEL_WEIGHT = '/home/frinks1/Downloads/DP/Heidelberg/bag_counting/yolov5l_training_results/training_backup_741_data_640ims_coco_customhype_adam/weights/best.pt'




# BLANK_IMAGE = np.zeros((750,1000, 3), dtype=np.float32)

### crop exp
RANDOM_NAME = ['23','r34','4e','sa','sf5','sdf2','awd','hfg','c5g','sfg','sef4','kj','vp','chmbt','rey','lil','fghgh','neo','34f']



### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx (frame_batch, model):
    # frame = [frame]
    # print(f"[INFO] Detecting. . . ")
    results = model(frame_batch, augment=True)

    # print( results.xyxyn)
    # print(len(results))
    # print(f"result type: {results}")
    # print(f"-------------------------XXX extracting labels and coordinates ---------------")
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])
    # print(results.pandas().xyxy[0])

    ### loopting through detections w.r.t image in order to create the final result_file

    batch_results = []

    for ir in range(len(results)):

        labels, cordinates = results.xyxyn[ir][:, -1], results.xyxyn[ir][:, :-1]

        batch_results.append((labels,cordinates))

    # print(f"len of batch_result: {len(batch_results)}")
    # print(batch_results)

    return batch_results


#### -------------------------------------- to save annotaions in txt file

def save_annotations(img, BBox,class_id, frame_name):
    
    xmin, ymin, xmax, ymax = BBox

    file_name = frame_name.split('/')[-1].split('.')[0]
    
    output_dir = OUTOUT_DIRR


    filex = open(f"{output_dir}/{file_name}.txt",'w')

    #### writing the annotaion in normalised yolo format ---> class_id x y w h 

    filex.write(f'{class_id} '+str((((xmin+xmax)/2)/img.shape[1]))+' '+str((((ymin+ymax)/2)/img.shape[0]))+' '+str(((xmax-xmin)/img.shape[1]))+' '+str(((ymax-ymin)/img.shape[0]))+'\n')

    filex.close()





### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def update_rects_plot_bbox(batch_results,imgs_rgb,classes, FRAME_COUNTER, vidc_name):

    # print(f"[INFO] Updating rects. . . ")


    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels

    """
    # print(imgs_rgb)
    imgs_results=[]
    rects_master = []
    x_mid = []


    output_dir  = OUTOUT_DIRR
    os.makedirs(output_dir, exist_ok= True)

    
    for im in range(len(imgs_rgb)):
        # image_h, image_w, _ = frame.shape  
        # 
         
        file_name = vidc_name.split('/')[-1].split('.')[0]
        filex = open(f"{output_dir}/{file_name}.txt",'w')


        
        frame = imgs_rgb[im]
        labels, cord = batch_results[im]
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        x_mid.append(x_shape//2)

        
        rects =[] ### rects per image


        # print(f"[INFO] Total {n} detections. . . ")
        # print(f"[INFO] Looping through all detections. . . ")


        ### looping through the detections per image
        for i in range(n):
            row = cord[i]
            if row[4] >= SCORE_THRESHOLD_BAG: ### threshold value for detection. We are discarding everything below this value
                # print(f"[INFO] Extracting BBox coordinates. . . ")
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
                # startX, startY, endX, endY = x1,y1,x2,y2

                class_id_int = int(labels[i])
                text_d = classes[int(labels[i])]

                cx = int((x1+x2)/2.0)
                cy = int((y1+y2)/2.0)

                xmin, ymin, xmax, ymax = x1, y1, x2, y2
                



                filex.write(f'{class_id_int} '+str((((xmin+xmax)/2)/frame.shape[1]))+' '+str((((ymin+ymax)/2)/frame.shape[0]))+' '+str(((xmax-xmin)/frame.shape[1]))+' '+str(((ymax-ymin)/frame.shape[0]))+'\n')



                ##### crop ratio 
                # xx1,yy1, xx2,yy2 = x1, y1, x2, y2 ### getting BBOx information for this particular ID
                # print(f"after xx1,yy1, xx2,yy2")
                # print(f" {xx1,yy1, xx2,yy2}")

                
                ### saving BBox as yolo annotation 
                # save_annotations(img = frame, BBox= [xx1,yy1,xx2,yy2], class_id = class_id_int, frame_name= vidc_name)

                # print(f"img shape : {img_master[z].shape}")
                # print(f"before ratio: {xx1,yy1,xx2,yy2}")



                # if ISRATIO:

                #     # if (centroidx[0] <= COUNT_ROIX_L2R):
                #     width = xx2 - xx1
                #     height = yy2 - yy1

                #     # ratiox = height / width
                #     # print(ratiox)

                #     # height = int(round(width * ratiox, 0))
                #     # print(height)

                #     top = (width - height) //2
                #     bottom = (width - height) - top
                #     yy1 = yy1 - top
                #     yy2 = yy2 + bottom

                #     print(width, height, top, bottom)

                #     print(xx1,yy1,xx2,yy2)



                #     img_crop = frame[yy1:yy2, xx1:xx2]
                #     area = width * height

                #     if (area) > 400000:
                #         # cv2.putText(img_crop, f"Area: {area}",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)


                #         # cv2.namedWindow("crop", cv2.WINDOW_NORMAL)
                #         # cv2.imshow('crop', img_crop)
                #         rvid_name = f"{random.choice(RANDOM_NAME)}_{vidc_name.split('/')[-1].split('.')[0]}"


                #         cv2.imwrite(f"./ratio_data/{rvid_name}_{FRAME_COUNTER}_{i}crop.jpg", img_crop)
                #         # cv2.imwrite(f"./ratio_data/{FRAME_COUNTER}_full.jpg", frame)
                # else:
                #     img_crop = frame[yy1:yy2, xx1:xx2]

                #     rvid_name = f"{random.choice(RANDOM_NAME)}_{vidc_name.split('/')[-1].split('.')[0]}"

                #     cv2.imwrite(f"./ratio_data/{rvid_name}_{FRAME_COUNTER}_{i}crop.jpg", img_crop)






                if text_d == 'cement_bag':

                    # print(f"----------- centroid before passing value for tracking: (cx,cy) ---> {(cx,cy)}")
                    if (cx> DETECT_ONLY_AFTER_CX) :#and (cy>DETECT_ONLY_AFTER_CY): #### to filter out wrong detection on person moving in the background
                        

                        # print(f"\nrects just after appending new detections to it:{rects}\n")

                        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) ## BBox
                        

                            
                        # cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

                        rects.append((x1,y1,x2,y2))

                    # cv2.imwrite(f"./result/{im}_final_output.jpg",frame)
                


            ## print(row[4], type(row[4]),int(row[4]), len(text_d))


        filex.close() ### closing the txt file

        imgs_results.append(frame)
        rects_master.append(rects)

    return imgs_results, rects_master, x_mid


################ --------    main area ------------------------


print(f"[INFO] Loading model... ")
## loading the custom trained model
# model =  torch.hub.load('ultralytics/yolov5', 'custom', path='bestm_label_bag.pt',force_reload=True) ## if you want to download the git repo and then run the detection
model =  torch.hub.load('./yolov5-master', 'custom', source ='local', path=BAG_MODEL_WEIGHT,force_reload=True) ### lastm_label_bag.pt--good result,  The repo is stored locally
model.conf = SCORE_THRESHOLD_BAG ### setting up confidence threshold
model.iou = IOU_THRESHOLD_BAG ## setting up iou threshold

classes = model.names ### class names in string format

# img_master=[]

#### looping through the images 

print(os.listdir(img_dir))
FRAME_COUNTER = 1

for i in os.listdir(img_dir):
    img_master=[]

    full_path = os.path.join(img_dir,i)
    vidc_name = full_path
    print(full_path)

    img = cv2.imread(full_path)
    img_master.append(img)
    # cv2.namedWindow("img",cv2.WINDOW_NORMAL)

    # while True:

    #     cv2.imshow("img", img)


    #     if cv2.waitKey(1) == ord("q"):
    #         break
    batch_results = detectx(frame_batch=img_master, model = model)

    img_master, rects_master, x_mid = update_rects_plot_bbox(batch_results=batch_results, imgs_rgb = img_master, classes= classes, FRAME_COUNTER= FRAME_COUNTER, vidc_name= vidc_name)

    FRAME_COUNTER+=1
