import os
import cv2
import json
import torch
import argparse
import numpy as np
from modules.utils.util import show_box
from modules.data.loader import ICDARDataLoader
from modules.utils.roi import batch_roi_transform
from modules.utils.converter import keys
from modules.utils.converter import StringLabelConverter
from modules.models.model import Recognizer
from ocr import ResizeNormalize
from PIL import Image
from torch.autograd import Variable


def test_roi_transform(config):
    train_loader = ICDARDataLoader(config).train()
    for batch_idx, gt in enumerate(train_loader):
        img_path, img, score, geo, train_mask, transcript, bbox, mapping = gt
        rois = batch_roi_transform(img, bbox[:, :8], mapping)
        print(rois.shape)
        rois = rois.permute(0, 2, 3, 1)
        for i in range(rois.shape[0]):
            print(transcript[i])
            roi = rois[i].detach().cpu().numpy()
            cv2.imshow('img', roi.astype(np.uint8))
            cv2.waitKey()

        break


def test_load_dataset(config):
    train_loader = ICDARDataLoader(config).train()
    for batch_idx, gt in enumerate(train_loader):
        img_path, img, score, geo, train_mask, transcript, bbox, mapping = gt
        img = img.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
        img = img[:, :, :, ::-1]
        ind = 0
        for img_index, box in zip(mapping, bbox):
            x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
            show_box(img[img_index], np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.int), transcript[ind])
            ind += 1
        print(bbox.shape)
        print(transcript.shape)
        print(mapping)
        print(img_path)

        break


def print_all_char():
    keys = ' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@!"#$%&\'[]()+,-./:;=?´ÉÈ'
    path = '/home/ishin/Development/east-mobilenet/datasets/train_gts'
    gt_path = list(sorted(os.listdir(path)))
    char = ''
    for pth in gt_path:
        gt_file = os.path.join(path, pth)
        with open(gt_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.replace('\xef\xbb\xbf', '')
            line = line.replace('\ufeff', '')
            gt = line.split(',')
            delim = ''
            label = ''
            for i in range(9, len(gt)):
                label += delim + gt[i]
                delim = ','
            label = label.strip()
            if not label.startswith("##"):
                for i in label:
                    if i in char or i in keys:
                        print(i)
                    else:
                        char += i

    # with open('modules/utils/key.txt', 'w') as fw:
    #     fw.write(char)
    print(char)


def icdar_2019():
    path = '../datasets/submit'
    out_path = '../datasets/detection'
    gt_path = list(sorted(os.listdir(path)))
    for pth in gt_path:
        gt_file = os.path.join(path, pth)
        output_path = os.path.join(out_path, pth)
        with open(gt_file, 'r') as f:
            lines = f.readlines()
        with open(output_path, 'w') as fw:
            for line in lines:
                line = line.replace('\xef\xbb\xbf', '')
                line = line.replace('\ufeff', '')
                gt = line.split(',')
                gt = gt[:9]
                fw.write('{0},{1},{2},{3},{4},{5},{6},{7},{8}\r\n'.format(
                    gt[0], gt[1], gt[2], gt[3], gt[4], gt[5], gt[6], gt[7], gt[8]
                ))


def ocr_demo(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='model path')
    parser.add_argument('-i', '--image_path', type=str, required=True, help='demo image path')
    args = parser.parse_args()

    model_path = args.model_path
    image_path = args.image_path

    # net init
    nclass = len(keys) + 1
    model = Recognizer(nclass, config)
    if torch.cuda.is_available():
        model = model.cuda()

    # load model
    print('loading pretrained model from %s' % model_path)
    model.load_state_dict(torch.load(model_path))

    converter = StringLabelConverter(keys)

    transformer = ResizeNormalize((180, 32))
    image = Image.open(image_path).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.LongTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))


if __name__ == '__main__':
    # config = json.load(open('config.json'))
    # test_load_dataset(config)
    # test_roi_transform(config)
    print_all_char()
    # ocr_demo(config)
    # icdar_2019()
