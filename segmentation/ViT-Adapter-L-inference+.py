from argparse import ArgumentParser

import mmcv

import mmcv_custom  # noqa: F401,F403
import mmseg_custom  # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
import cv2
import os.path as osp
import os
import time
from scipy.io import loadmat
import numpy as np
import csv
import json

parser = ArgumentParser()

parser.add_argument('--imgs', help='Image file')
parser.add_argument('--out', help='out dir')
parser.add_argument('--colorcsv', help='filepath of object150_info.csv')
parser.add_argument('--colormat', help='filepath of color150.mat')
parser.add_argument('--mode', help='choose from "svi"or""')

parser.add_argument(
    '--config',
    default='configs/ade20k/mask2former_beitv2_adapter_large_896_80k_ade20k_ss.py',
    help='Config file')
parser.add_argument(
    '--checkpoint',
    default='mask2former_beitv2_adapter_large_896_80k_ade20k.pth',
    help='Checkpoint file')
parser.add_argument(
    '--device',
    default='cuda:0',
    help='Device used for inference')
parser.add_argument(
    '--palette',
    default='ade20k',
    help='Color palette used for segmentation map')
parser.add_argument(
    '--opacity',
    type=float,
    default=1,
    help='Opacity of painted segmentation map. In (0, 1] range.')
args = parser.parse_args()
'''
args = parser.parse_args(        
    "--imgs /kaggle/input/hailunshi-1024x2048-10 "
    "--out /kaggle/working/hailunshi-1024x2048-10 "
    "--mode svi "
    "--colorcsv /kaggle/input/notebook4seg/ade20k/object150_info.csv "
    "--colormat /kaggle/input/notebook4seg/ade20k/color150.mat".split()
    )
'''

# prepare files and create dicts

names = {}
with open(args.colorcsv) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

colors = loadmat(args.colormat)['colors']  # 标签对应的颜色 numpy.ndarray [[R,G,B][R,G,B]……]

result2save = {}


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def visualize_result(pred, img_name):
    # print predictions in descending order
    unique_ratio = []
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)  # 统计独特的数值，及其出现的次数
    for idx in np.argsort(counts)[::-1]:  # 将count列表中的元素值从大到小排列，给出对应元素的序号
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        # if ratio > 0.1:
        print("  {}: {:.2f}%".format(name, ratio))
        unique_ratio.append([default_dump(name), round(default_dump(ratio), 2)])
    result2save[img_name] = unique_ratio


def single_img_inference(model, img_path, out_folder, palette, opacity):
    # test a single image
    result = inference_segmentor(model, img_path)
    file_name = osp.basename(img_path)
    visualize_result(result, file_name)
    out_name= osp.splitext(osp.basename(img_path))[0]
    mmcv.mkdir_or_exist(out_folder)
    """
    # show the results
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img_path, result,
                            palette=get_palette(palette),
                            show=False, opacity=opacity)
    out_png_name =  out_name + ".png"
    out_png_path = osp.join(out_folder, out_png_name)
    cv2.imwrite(out_png_path, img)
    """
    out_npy_name = out_name + ".npy"
    out_npy_path = osp.join(out_folder, out_npy_name)
    result_np = np.uint8(result[0]) + 1
    np.save(out_npy_path, result_np)
    # cv2.imwrite(out_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])  无损png
    print(f"Result is save at {out_folder}")
    # save json
    json_fp = osp.join(out_folder, "ImgDict.json")
    with open(json_fp, 'w') as f:
        json.dump(result2save, f, indent=4)


def make_imgs_list(input_str):
    """make a list of image(s).

  Args:
      input_str (file[str] or filepath[str]): Either a image file 
      or a folder with images in it.

  Returns:
      (imgs_l[list]): The paths of images.
  """
    img_l = []

    if osp.isdir(input_str):
        input_files = [os.path.join(input_str, f) for f in os.listdir(input_str)]

    elif osp.isfile(input_str):
        input_files = [input_str]
    else:
        raise AssertionError("the input is not a file or folder")

    for f in input_files:
        if f.endswith(('jpg', 'png', 'jpeg', 'bmp')):
            img_l.append(f)

    return img_l


def time_cal(start, now, temp, idx, tot, name):
    predict = (float(tot) - idx - 1) * ((now - start) / (idx + 1))
    predict_h = int(predict // 3600)
    predict_m = int((predict - (predict // 3600) * 3600) // 60)
    predict_s = int(predict - (predict_h * 3600) - (predict_m * 60))
    last = now - start
    last_h = int(last // 3600)
    last_m = int((last - (last // 3600) * 3600) // 60)
    last_s = int(last - (last_h * 3600) - (last_m * 60))
    print(
        f'当前总用时：{last_h}h{last_m}m{last_s}s '
        f'本张用时：{round(now - temp, 2)}s '
        f'平均单张用时：{round((now - start) / (idx + 1), 2)}s '
        f'预计剩余用时：{predict_h}h{predict_m}m{predict_s}s'
    )


def main():
    # build the model from a config file and a checkpoint file

    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)

    # inference images

    imgs_list = make_imgs_list(args.imgs)
    if args.mode == "svi":
        imgs_list.sort(key=lambda x: int(osp.basename(x).split("_")[0]))

    idx = 0
    tot = len(imgs_list)
    start = time.time()
    temp = start
    for img in imgs_list:
        print(f'开始处理：{idx + 1}/{tot} {img}')
        single_img_inference(model, img, args.out, args.palette, args.opacity)  # 推理图像
        now = time.time()
        time_cal(start, now, temp, idx, tot, img)  # 打印用时信息
        idx += 1
        temp = now


if __name__ == '__main__':
    main()
