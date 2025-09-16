# @Author: CXY
import pandas as pd
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import io
import argparse
import yaml
from model import Repetitive_pose, Mction_trigger
import csv

plt.rcParams['font.family'] = 'Times New Roman'
torch.multiprocessing.set_sharing_strategy('file_system')


def normalize_landmarks(all_landmarks):
    x_max = np.expand_dims(np.max(all_landmarks[:, :, 0], axis=1), 1)
    x_min = np.expand_dims(np.min(all_landmarks[:, :, 0], axis=1), 1)

    y_max = np.expand_dims(np.max(all_landmarks[:, :, 1], axis=1), 1)
    y_min = np.expand_dims(np.min(all_landmarks[:, :, 1], axis=1), 1)

    z_max = np.expand_dims(np.max(all_landmarks[:, :, 2], axis=1), 1)
    z_min = np.expand_dims(np.min(all_landmarks[:, :, 2], axis=1), 1)

    all_landmarks[:, :, 0] = (all_landmarks[:, :, 0] - x_min) / (x_max - x_min)
    all_landmarks[:, :, 1] = (all_landmarks[:, :, 1] - y_min) / (y_max - y_min)
    all_landmarks[:, :, 2] = (all_landmarks[:, :, 2] - z_min) / (z_max - z_min)

    all_landmarks = all_landmarks.reshape(len(all_landmarks), -1)
    return all_landmarks


def show_image(img, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()


class PoseClassificationVisualizer(object):
    def __init__(self,
                 class_name,
                 plot_location_x=0.05,
                 plot_location_y=0.05,
                 plot_max_width=0.4,
                 plot_max_height=0.4,
                 plot_figsize=(16, 8),
                 plot_x_max=None,
                 plot_y_max=None,
                 counter_location_x=0.85,
                 counter_location_y=0.05,
                 counter_font_color='red',
                 counter_font_size=0.15):
        self._class_name = class_name
        self._plot_location_x = plot_location_x
        self._plot_location_y = plot_location_y
        self._plot_max_width = plot_max_width
        self._plot_max_height = plot_max_height
        self._plot_figsize = plot_figsize
        self._plot_x_max = plot_x_max
        self._plot_y_max = plot_y_max
        self._counter_location_x = counter_location_x
        self._counter_location_y = counter_location_y
        self._counter_font_color = counter_font_color
        self._counter_font_size = counter_font_size
        self._counter_font = None
        self._pose_classification_history = []
        self._pose_classification_filtered_history = []

    def __call__(self,
                 frame,
                 pose_classification,
                 pose_classification_filtered,
                 repetitions_count):
        self._pose_classification_history.append(pose_classification)
        self._pose_classification_filtered_history.append(pose_classification_filtered)
        output_img = Image.fromarray(frame)

        output_width = output_img.size[0]
        output_height = output_img.size[1]

        img, img_copy, y = self._plot_classification_history(output_width, output_height)

        img.thumbnail((640,640),Image.LANCZOS)

        output_img.paste(img,
                         (int(output_width * self._plot_location_x),
                          int(output_height * self._plot_location_y)))

        output_img_draw = ImageDraw.Draw(output_img)
        if self._counter_font is None:
            font_size = int(output_height * self._counter_font_size)
           
        output_img_draw.text((output_width * self._counter_location_x,
                              output_height * self._counter_location_y),
                             str(repetitions_count),
                             font=self._counter_font,
                             fill=self._counter_font_color)

        return img_copy, output_img, y

    def _plot_classification_history(self, output_width, output_height):
        fig = plt.figure(figsize=self._plot_figsize)
        for classification_history in [self._pose_classification_history,
                                       self._pose_classification_filtered_history]:
            y = []
            for classification in classification_history:
                if classification is None:
                    y.append(None)
                else:
                    y.append(classification)
            # print(f'y is {y}')

            plt.plot(y, 'o-', markersize=5, color=(228/255, 23/255, 73/255)) #linewidth=5,

        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Frame',fontsize=18)
        plt.ylabel('Score ',fontsize=18)
        plt.title('RepCount of `{}`'.format(self._class_name),fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)  # 主刻度

        if self._plot_y_max is not None:
            plt.ylim(top=self._plot_y_max)
        if self._plot_x_max is not None:
            plt.xlim(right=self._plot_x_max)
        img_copy = fig
        # Convert plot to image.
        buf = io.BytesIO()
        dpi = min(
            output_width * self._plot_max_width / float(self._plot_figsize[0]),
            output_height * self._plot_max_height / float(self._plot_figsize[1]))
        fig.savefig(buf, dpi=dpi)
        buf.seek(0)
        img = Image.open(buf)
        plt.close()

        return img, img_copy, y


def main(args):
    one_video_name = 'my_full_6_average_angles'
    save_frame_c = False
    save_skeleton_c = False
    save_density_c = True
    density_save_path = 'test_density_img_6_ave_angles'
    save_video_c = True
    if os.path.isfile(args.config):
        with open(args.config, "r") as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
    else:
        raise ValueError("Config file does not exist.")

    csv_label_path = config['dataset']['csv_label_path']
    root_dir = config['dataset']['dataset_root_dir']

    output_video_dir = os.path.join(root_dir, 'video_visual_output_phd', 'test_' + one_video_name)

    input_video_dir = os.path.join(root_dir, 'video', 'test')  # test

    poses_save_dir = os.path.join(root_dir, 'test_poses_6_ave_phd')
    

    if not os.path.isdir(output_video_dir):
        os.makedirs(output_video_dir)

    test_csv_name = os.path.join(root_dir, 'annotation', 'test.csv')
    test_df = pd.read_csv(test_csv_name)

    label_pd = pd.read_csv(csv_label_path)
    index2action = {}
    length_label = len(label_pd.index)
    for label_i in range(length_label):
        one_data = label_pd.iloc[label_i]
        action = one_data['action']
        label = one_data['label']
        index2action[label] = action

    num_classes = len(index2action)
    

    # load model pth
    model = Repetitive_pose(None, None, None, None, dim=config['Repetitive_pose']['dim'], heads=config['Repetitive_pose']['heads'],
                    enc_layer=config['Repetitive_pose']['enc_layer'], learning_rate=config['Repetitive_pose']['learning_rate'],
                    seed=config['Repetitive_pose']['seed'], num_classes=num_classes, alpha=config['Repetitive_pose']['alpha'])
    weight_path = args.pth
    new_weights = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(new_weights)
    model.eval()

    enter_threshold = config['Mction_trigger']['enter_threshold']
    exit_threshold = config['Mction_trigger']['exit_threshold']
    momentum = config['Mction_trigger']['momentum']

    # landmark specific parameters initialization
    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(217, 83, 79), thickness=2, circle_radius=2)
    connection_drawing_spec = mp_drawing.DrawingSpec(color=(8, 255, 200), thickness=2, circle_radius=2)

    for i in range(0, len(test_df)):
        video_name = test_df.loc[i, 'name']
        print(video_name)
        gt_count = test_df.loc[i, 'count']

        poses_save_path = os.path.join(poses_save_dir, video_name.replace('mp4', 'npy'))

        all_landmarks = np.load(poses_save_path).reshape(-1, 105) # 33*3 coordinates +6 joint angles
        all_landmarks_tensor = torch.from_numpy(all_landmarks).float()
        # print(all_landmarks_tensor.shape)
        all_output = torch.sigmoid(model(all_landmarks_tensor))
        
        print('all_output: {} '.format(all_output.shape) )
        # 将输出转换为 NumPy 数组
        all_output_numpy = all_output.detach().cpu().numpy()

        # 保存到 TXT 文件
        output_file_path = 'phd_2output.txt'
        np.savetxt(output_file_path, all_output_numpy, fmt='%.6f')  # 选择合适的格式

        best_mae = float('inf')
        real_action = 'none'
        real_index = -1
        

        # 初始化动作计数字典
        action_counts = {action_type: 0 for action_type in index2action.values()}
        

        for index in index2action:
            action_type = index2action[index]
            print(action_counts)
            
         
        # Initialize action trigger.
            repetition_salient_1 = Mction_trigger(
            action_name=action_type,
            enter_threshold=enter_threshold,
            exit_threshold=exit_threshold)
            repetition_salient_2 = Mction_trigger(
            action_name=action_type,
            enter_threshold=enter_threshold,
            exit_threshold=exit_threshold)
            

            classify_prob = 0.5
            curr_pose = 'holder'
            init_pose = 'pose_holder'
            print(all_output.shape)

# 打开文件以写入模式
            with open('Mction_trigger_log_phd.txt', 'a') as f:
                for frame_index, output in enumerate(all_output):
                    output_numpy = output[index].detach().cpu().numpy()
                    classify_prob = output_numpy * (1. - momentum) + momentum * classify_prob
                    salient1_triggered = repetition_salient_1(classify_prob)
                    reverse_classify_prob = 1 - classify_prob
                    salient2_triggered = repetition_salient_2(reverse_classify_prob)

                    
                    # 处理初始姿态
                    if init_pose == 'pose_holder':
                        if salient1_triggered:
                            init_pose = 'salient1'
                        elif salient2_triggered:
                            init_pose = 'salient2'

                    # 更新当前姿态和计数
                    if init_pose == 'salient1':
                        if curr_pose == 'salient1' and salient2_triggered:
                            action_counts[action_type] += 1  # 统计次数
                            log_entry = f'触发动作: {action_type}, 次数: {action_counts[action_type]}, 帧索引: {frame_index}\n'
                            f.write(log_entry)  # 写入文件
                            print(log_entry.strip())  # 可选：输出到控制台
                    else:
                        if curr_pose == 'salient2' and salient1_triggered:
                            action_counts[action_type] += 1  # 统计次数
                            log_entry = f'触发动作: {action_type}, 次数: {action_counts[action_type]}, 帧索引: {frame_index}\n'
                            f.write(log_entry)  # 写入文件
                            print(log_entry.strip())  # 可选：输出到控制台

                    # 更新当前姿态
                    if salient1_triggered:
                        curr_pose = 'salient1'
                    elif salient2_triggered:
                        curr_pose = 'salient2'


                    



       

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate our Repetitive_pose')
    parser.add_argument('--config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('--pth', type=str, metavar='DIR',
                        help='path to a checkpoint')
    args = parser.parse_args()
    main(args)
