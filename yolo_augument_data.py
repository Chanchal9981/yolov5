import albumentations as A
import cv2
import os
import random
import tqdm


# Initialising augumentation
transform = A.Compose([
    # A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1)
], bbox_params=A.BboxParams(format='yolo'))


img_dir = "/home/frinks1/Meraj/PPE/yolov5_ppe_/training/data/"
label_dir = "/home/frinks1/Meraj/PPE/yolov5_ppe_/training/data_label/"
labels = ["helmet","vest","boot"]

def create_bbox(file_path):
    f = open(file_path, "r")
    lines = f.readlines()
    # print(lines)
    bboxes = []
    classes = []
    for line in lines:
        classes.append(int(line.split(" ")[0]))
        bboxes.append([float(i) for i in line.split(" ")[1:]])
    class_names = [labels[class_id] for class_id in classes]
    for i in range(len(bboxes)):
        bboxes[i].append(class_names[i])
    # print("Actual: ", bboxes)
    return bboxes

def create_txt_file(bboxes, img_name, img_count):
    class_names = [box[-1] for box in bboxes]
    class_ids = [labels.index(class_name) for class_name in class_names]
    bboxes = [[round(val, 6) for val in box[:-1]] for box in bboxes]
    # print(class_ids, bboxes)
    save_path = os.path.join("/home/frinks1/Meraj/PPE/yolov5_ppe_/training/data_label/augumented/", f"{img_name.split('.')[0]}_augumented{img_count}.txt")
    # print("Save:" , save_path)
    f = open(save_path, "w+")
    for bbox, class_id in zip(bboxes, class_ids):
        line = f"{class_id} "+" ".join([str(val) for val in bbox]) + "\n"
        f.write(line)
        # print(line, sep="")
    f.close()

original_img = os.listdir(img_dir)
i=0
img_count = 1
while i < 2:
    for img_name in tqdm.tqdm(original_img):
        # print(img_name)
        try:
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(label_dir, ".".join(img_name.split(".")[:-1])+".txt")
            img = cv2.imread(img_path)
            bboxes = create_bbox(label_path)
            transformed = transform(image=img, bboxes=bboxes)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            # Saving img and labels
            img_save_path = os.path.join(img_dir, f"{img_name.split('.')[0]}_augumented{img_count}.jpg")
            cv2.imwrite(img_save_path, transformed_image)
            create_txt_file(transformed_bboxes, img_name, img_count)
            img_count += 1
            # exit()
            # transformed_class_labels = transformed['class_labels']
            # print(transformed_bboxes)
            # exit()
            # print(transformed_class_labels)
        except Exception as e:
            print(img_name)
            print(e)
            continue

    i += 1

