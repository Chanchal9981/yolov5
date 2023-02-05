import os
import cv2
from tqdm import tqdm

img_folder = "./vid2frame_no"
vid_out = "./output"


def frame2video(img_folder, vid_out):

    print(f"--------------- PROCESS STARTED ----------------")
    width = 2560
    height = 1440
    fps = 8
    os.makedirs(vid_out, exist_ok=True)

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(f"{vid_out}/notagfinal.mp4",codec,fps,(width,height))
    print(os.listdir(img_folder))

    img_list =  sorted(os.listdir(img_folder), key=lambda x: int(x.split(".")[0]))
    # img_list =  sorted(os.listdir(img_folder), key=lambda x: x.split(".")[0])

    print(img_list)


    for i_frame in tqdm(img_list):

        framex = cv2.imread(os.path.join(img_folder,i_frame))

        print(f"---------- writing frame: {i_frame}")
        writer.write(framex)
    
    writer.release()
    print(f"Video created!!!")


frame2video(img_folder=img_folder, vid_out=vid_out)
