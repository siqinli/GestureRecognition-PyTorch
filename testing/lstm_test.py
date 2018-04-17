'''
	The online testing code of finetune CNN+LSTM network.

	Siqin Li 	
	April, 2018
'''

import argparse
import os
import shutil
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
from torch.autograd import Variable 
import PIL.Image as Image

import sys
sys.path.insert(0, '../training')
from lstm_arch import *

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('model', help='path to the pretrained model')
parser.add_argument('--frames_dir', default=None, help='video frames to classify (default: webcam)')
parser.add_argument('--freq', type=int, default=20, help='classify frequency (length of subsequence)')
parser.add_argument('--flow_model', help='path to the trained flow mode')
parser.add_argument('--output', type=str, default=None, help='save the video frames to output directory')

classes = ['HoldHover', 'Land', 'LiftOff', 'MoveDownward', 
	'MoveForward', 'MoveLeft', 'MoveRight', 'MoveUpward', 'ReleaseSlingLoad']

H = 20
L = -20
WAIT_TIME = 60

save_path = "/home/siqinli/droneGesture/results"



def cal_flow(frames, transform=None):
	'''
	compute optical flow, return CV image list
	input: 
			frames: list of frame
			transform: Torch transform
	return:
			optical flow as a torch tensor ((num_frames - 1) * 3 * l * w)) 

	'''
	flow_frames = []

	prvs = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
	hsv = np.zeros_like(frames[0])
	hsv[..., 1] = 255

	for idx in range(1, len(frames)):
		nxt = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
		flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.1, 0)
		flow = flow_map(flow)
		x_flow = transform(Image.fromarray(np.uint8(flow[:, :, 0])))
		y_flow = transform(Image.fromarray(np.uint8(flow[:, :, 1])))
		flow = torch.cat((x_flow, y_flow, torch.zeros(x_flow.size())), dim=0)
		flow_frames.append(flow)
		prvs = nxt

	flow_frames = torch.stack(flow_frames)
	return flow_frames

def flow_map(x):
	# normalize flow to 0 - 255
	y = 255 * (x - L) / (H - L)
	y = np.maximum(0, y)
	y = np.minimum(255, y)
	return y

def test(input, model, eval_ratio=0.5, inflow=None, f_model=None):
	eval_last_len = int(len(input) * eval_ratio)

	model.eval()
	input_var = Variable(input)
	input_var = input_var.cuda()
	outputs, _ = model(input_var)
	weight = Variable(torch.Tensor(range(outputs.shape[0])) / (outputs.shape[0] - 1) * 2).cuda()
	output = torch.mean(outputs * weight.unsqueeze(1), dim=0)
	output = nn.functional.softmax(output, dim=0)
	print output

	if inflow is not None and f_model is not None:
		f_model.eval()
		inflow_var = Variable(inflow)
		inflow_var = inflow_var.cuda()
		outflows, _ = f_model(inflow_var)
		outflow = torch.mean(outflows[-eval_last_len:], dim=0)
		outflow = nn.functional.softmax(outflow, dim=0)
		output += outflow

	confidence, idx = torch.max(output.data.cpu(), 0)

	return confidence.numpy()[0], idx.numpy()[0]


def main():
	global args
	args = parser.parse_args()

	if args.output is not None:
		if os.path.exists(args.output):
			shutil.rmtree(args.output)
		os.mkdir(args.output)

	model_info = torch.load(args.model)
	# print model_info
	num_classes = 9

	print 'LSTM using pretrained model ' + model_info['arch']

	original_model = models.__dict__[model_info['arch']](pretrained=False)	
	model = FineTuneLstmModel(original_model, model_info['arch'],
			num_classes, model_info['lstm_layers'], model_info['hidden_size'], model_info['fc_size'])
	model.cuda()
	model.load_state_dict(model_info['state_dict'])
	# print model_info['best_prec1']
	# print (model_info['lstm_layers'], model_info['hidden_size'], model_info['fc_size'])
	# print list(model.rnn.state_dict()) #.numpy()
	# print model.rnn.bias_ih_l0.data
	# print model.rnn.bias_hh_l0.data
	
	if args.flow_model is not None:
		print 'Loding optical model...'
		f_model_info = torch.load(args.flow_model)
		f_original_model = models.__dict__[f_model_info['arch']](pretrained=False)
		f_model = FineTuneLstmModel(f_original_model, f_model_info['arch'],
			num_classes, f_model_info['lstm_layers'], f_model_info['hidden_size'], f_model_info['fc_size'])
		f_model.cuda()
		f_model.load_state_dict(f_model_info['state_dict'])

	# data preprocessing...
	tran = transforms.Compose([transforms.Resize(224),
								transforms.CenterCrop(224),
								transforms.ToTensor(),
								transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.339, 0.224, 0.225])])

	if args.frames_dir is not None:
		# classify a video 
		frames_list = sorted(os.listdir(args.frames_dir))
		sublist = []
		for idx in range(len(frames_list)):
			sublist.append(frames_list[idx])
			
			if len(sublist) == args.freq and idx < len(frames_list)-1:
				frames = []
				for f in sublist:
					frame = Image.open(args.frames_dir + '/' + f)
					frame = tran(frame)
					frames.append(frame)
				frames = torch.stack(frames)
				frames = frames[:-1] - frames[1:]

				print 'classifying...'
				confidence, label = test(frames, model)
				print confidence, label
				print classes[label], confidence
				sublist = []
	else:
		# read frames from webcam
		cap = cv2.VideoCapture(0)
		frames = []
		raw_frames = []
		text = None
		label_list = []
		lbl = 1
		idx = 0

		# text settings
		font = cv2.FONT_HERSHEY_SIMPLEX
		bottomLeftCornerOfText = (10,30)
		fontScale = 1
		fontColor = (255,255,255)
		lineType = 2

		_, pre_raw_frame = cap.read()
		flag = False
		# wait_count = WAIT_TIME
		# count = 0
		while(True):
			# Capture frame-by-frame
			# if wait_count == 0:
			ret, raw_frame = cap.read()
			# 	wait_count += 1

			# else:

			

			# Our operations on the frame come here

			frame = tran(Image.fromarray(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)))

			if np.mean((raw_frame - pre_raw_frame) ** 2) > 0.01:
				flag = True
			else:
				flag = False

			pre_raw_frame = raw_frame	


			if flag or len(frames) > 0:
				frames.append(frame)
				raw_frames.append(raw_frame)
			# elif len(frames) > 1:
			if len(frames) == args.freq:
				frames = torch.stack(frames)
				frames = frames[:-1] - frames[1:]

				print 'classifying...'
				if args.flow_model is None:
					confidence, label = test(frames, model)
				else:
					flow_frames = cal_flow(raw_frames, tran)
					confidence, label = test(frames, model, inflow=flow_frames, f_model=f_model)

				if confidence > 0.9:
					text = classes[label] + ": " + str(confidence)
					lbl = label
				else:
					text = None
				frames = []
				raw_frames = []

			label_list.append(lbl)

			# Display the resulting frame

			cv2.putText(raw_frame, text,
				bottomLeftCornerOfText, 
				font, 
				fontScale,
				fontColor,
				lineType)
			cv2.imshow('frame',raw_frame)

			if args.output is not None:
				cv2.imwrite(os.path.join(args.output, "%05d.jpg" % idx), raw_frame)

			idx += 1
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		# When everything done, release the capture
		if args.output is not None:
			np.save(os.path.join(args.output, 'label_list.npy'), label_list)
		print "stop"
		cap.release()
		cv2.destroyAllWindows()


if __name__ == '__main__':
	main()