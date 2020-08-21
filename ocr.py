import os
import cv2
import torch
import numpy as np
import argparse
import logging
import torch.utils.data as torchdata
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
from PIL import Image
from modules.utils.converter import keys
from modules.models.model import Recognizer
from modules.models.loss import OCRLoss
from modules.utils.util import make_dir
from modules.utils.converter import StringLabelConverter

logging.basicConfig(level=logging.DEBUG, format='')


class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        # img.sub_(0.5).div_(0.5)
        return img


class AlignCollate(object):

    def __init__(self, img_h=32, img_w=180):
        self.imgH = img_h
        self.imgW = img_w

    def __call__(self, batch):
        images, labels = zip(*batch)
        transform = ResizeNormalize((self.imgW, self.imgH))
        images = [transform(image) for image in images]
        images = torch.stack(images, 0)
        # print(images)

        return images, labels


class OCRDataset(Dataset):
    def __init__(self, gt_file_path=None):
        self.gt_file_path = gt_file_path
        self.img_path_list, self.label_list = self.__read_data()

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        label = self.label_list[index]
        try:
            # img = Image.open(img_path).convert('L')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = Image.fromarray(np.uint8(img))
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]
        return img, label

    def __read_data(self):
        base = os.path.dirname(self.gt_file_path)
        image_path_list = []
        label_list = []
        with open(self.gt_file_path) as f:
            lines = f.readlines()
            for line in lines:
                gt = line.strip().split(',')
                delim, label = '', ''
                for i in range(2, len(gt)):
                    label += delim + gt[i]
                    delim = ','
                image_path_list.append(os.path.join(base, gt[0].strip()))
                label_list.append(label.strip().strip('\"'))

        return image_path_list, label_list


class OCRDataLoader:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_workers = args.workers
        self.__ocr_dataset = OCRDataset(args.gt_file)
        self.__collate_fn = AlignCollate()

    def train(self):
        trainLoader = torchdata.DataLoader(self.__ocr_dataset, num_workers=self.num_workers, batch_size=self.batch_size,
                                           shuffle=True, collate_fn=self.__collate_fn)
        return trainLoader


def train_epoch(model, dataloader, optimizer, criterion, epoch, converter, device):
    model.train()
    total_loss = 0
    num_correct = 0
    for batch_idx, data in enumerate(dataloader):
        img, transcript = data
        img = Variable(img.to(device))

        optimizer.zero_grad()

        preds = model(img)
        preds_size = torch.LongTensor([preds.size(0)] * int(preds.size(1))).to(device)
        pred = (preds, preds_size)

        labels, label_lengths = converter.encode(transcript)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        gt = (labels, label_lengths)

        loss = criterion(gt, pred)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)

        n_correct = 0
        for pred, target in zip(sim_preds, transcript):
            if pred == target:
                n_correct += 1

        num_correct += n_correct / dataloader.batch_size

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] | Batch Loss: {:.8f}'.format(
                epoch,
                batch_idx * dataloader.batch_size,
                len(dataloader) * dataloader.batch_size,
                100.0 * batch_idx / len(dataloader),
                loss.item()
            ))

    log = {
        'loss': total_loss / len(dataloader),
        'correct': num_correct / len(dataloader)
    }
    return log


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MLT-OCR')
    parser.add_argument('-b', '--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('-e', '--epochs', default=500, type=int, help='number of epoch')
    parser.add_argument('-f', '--gt_file', default='../datasets/words/gt.txt', type=str, help='path to gt.txt file')
    parser.add_argument('-l', '--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('-p', '--checkpoint', default='pretrained/', type=str, help='save path')
    parser.add_argument('-w', '--workers', default=5, type=int, help='number of workers')
    args = parser.parse_args()

    make_dir(args.checkpoint)

    converter = StringLabelConverter(keys)

    # check whether cuda or cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    train_dataloader = OCRDataLoader(args).train()

    # init model
    num_class = len(keys) + 1
    model = Recognizer(num_class, None)
    model.summary()
    model = model.to(device)

    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # init loss
    loss = OCRLoss()

    # train
    start_epoch = 1
    for epoch in range(start_epoch, args.epochs + 1):
        log = train_epoch(model, train_dataloader, optimizer, loss, epoch, converter, device)
        print(log)
        filename = os.path.join(args.checkpoint, 'ocr-epoch{:03d}-loss-{:.4f}.pth.tar'.format(epoch, log['loss']))
        torch.save(model.state_dict(), filename)
