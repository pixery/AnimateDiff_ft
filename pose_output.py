
from controlnet_aux import OpenposeDetector
import os
import PIL.Image
import imageio
import decord
decord.bridge.set_bridge('torch')

input_video = "/home/avcr/Desktop/cem/AnimateDiff_ft/controlnet_video_processed/data/zoom_in.mp4"
output_video = "pose_face_zoom_in.mp4"
output_folder = "temp"







def generate_pose(input_video, output_folder, output_video, pose_model):
    include_hand = False
    include_face = True
    is_video = True
    if not input_video.endswith(".mp4"):
        is_video = False

    if is_video:
        vr = decord.VideoReader(input_video, width=512, height=512)

        video = vr.get_batch(list(range(16)))
        poses = [pose_model(v, include_body = False, include_face=include_face, include_hand = include_hand) for v in video]
    else:
        poses = [pose_model(PIL.Image.open(input_video), include_face=include_face, include_hand = include_hand)] * 16


    for i,p in enumerate(poses):
        p.save(f'{output_folder}/outputs/frames/{i}.png')

    os.system(f'/home/avcr/Desktop/cem/animatediff_finetune/AnimateDiff/ffmpeg-git-20230915-amd64-static/ffmpeg -r 8 -i {output_folder}/outputs/frames/%d.png -c:v libx264 -vf fps=8 {output_folder}/pose_data/{output_video}')




if __name__ == "__main__":
    output_folder =  "/home/avcr/Desktop/cem/animatediff_finetune/AnimateDiff/wink_dataset_fixed_processed_cleaned/"
    path = "data/"
    output_path = "/home/avcr/Desktop/cem/animatediff_finetune/AnimateDiff/wink_dataset_fixed_processed_cleaned/pose_data/"
    pose_model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    ind = 0
    for i in os.listdir(output_folder + path):
        print (f"Current File: {i}, {ind}")
        ind += 1
        if i.endswith(".mp4") or i.endswith(".jpg") or i.endswith(".png"):
            generate_pose(output_folder + path + i, output_folder, "pose_" + i, pose_model)