'''
	Data loader

	Siqin Li
	April, 2018
'''


import os
import torch
import numpy as np
import cv2
import PIL.Image as Image

from torch.utils.data import Dataset
from torchvision import transforms, utils

class CLMarshallingDataset(Dataset):
	def __init__(self, root_dir, transform=None):
		'''
		structure of root_dir: 'root_dir/class_i/video_i/img_i.jpg'
		'''
		self.root_dir = root_dir
		self.transform = transform
		self.classes = sorted(os.listdir(self.root_dir))
		self.count = [len(os.listdir(self.root_dir + '/' + c)) for c in self.classes]
		self.acc_count = [self.count[0]]
		for i in range(1, len(self.count)):
				self.acc_count.append(self.acc_count[i-1] + self.count[i])
		# self.acc_count = [self.count[i] + self.acc_count[i-1] for i in range(1, len(self.count))]


	def cal_flow(self, frames, transform):
		'''
		compute optical flow, return PIL image list
		'''
		flow_frames = []

		prvs = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
		hsv = np.zeros_like(frames[0])
		hsv[..., 1] = 255

		for idx in range(1, len(frames)):
			nxt = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
			flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.1, 0)
			mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
			hsv[...,0] = ang * 180 / np.pi / 2
			hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
			bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

			pil_im = Image.fromarray(bgr)
			pil_im = transform(pil_im)
			flow_frames.append(pil_im)
			prvs = nxt

		return flow_frames


	def __len__(self):
		l = np.sum(np.array([len(os.listdir(self.root_dir + '/' + c)) for c in self.classes]))
		return l

	def __getitem__(self, idx):
		for i in range(len(self.acc_count)):
			if idx < self.acc_count[i]:
				label = i
				break

		class_path = self.root_dir + '/' + self.classes[label] 
		class_path_flow = self.root_dir + '_flow/' + self.classes[label]

		if label:
			file_path = class_path + '/' + sorted(os.listdir(class_path))[idx-self.acc_count[label]]
			flow_path = class_path_flow + '/' + sorted(os.listdir(class_path))[idx-self.acc_count[label]]
		else:
			file_path = class_path + '/' + sorted(os.listdir(class_path))[idx]
			flow_path = class_path_flow + '/' + sorted(os.listdir(class_path))[idx]

		frames = []
		flow_frames = []

		# print os.listdir(file_path)
		file_list = sorted(os.listdir(file_path))
		# print file_list
		flow_list = sorted(os.listdir(flow_path))
		# v: maximum translation in every step
		v = 2
		offset = 0
		for i, f in enumerate(file_list):
			frame = Image.open(file_path + '/' + f)
			#translation
			offset += random.randrange(-v, v)
			offset = min(offset, 3 * v)
			offset = max(offset, -3 * v)
			frame = frame.transform(frame.size, Image.AFFINE, (1, 0, offset, 0, 1, 0))
			if self.transform is not None:
				frame = self.transform[0](frame)
			frames.append(frame)

			if i < len(file_list) - 1:
				flow_frame_x = Image.open(flow_path + '/x_' + f)
				flow_frame_y = Image.open(flow_path + '/y_' + f)
				if self.transform is not None:
					flow_frame_x = self.transform[1](flow_frame_x)
					flow_frame_y = self.transform[1](flow_frame_y)
				flow_frame = torch.cat((flow_frame_x, flow_frame_y, torch.zeros(flow_frame_x.size())), dim=0)
				flow_frames.append(flow_frame)

		frames = torch.stack(frames)
		frames = frames[: -1] - frames[1:]

		flow_frames = torch.stack(flow_frames)

		return frames, label, flow_frames
