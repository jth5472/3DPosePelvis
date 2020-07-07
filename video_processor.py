

import cv2
from tf_pose import common
from tf_pose.common import MPIIPart
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import model_wh,get_graph_path
import logging
import argparse
import os
import time
from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import imageio
import copy
from sklearn.preprocessing import MinMaxScaler
import math

vicon_mappings = { "LTHI": 2,
				   "LKNE":3,
				   "LANK":5,
				   "RTHI":10,
				   "RKNE":11,
				   "RANK":13,
}

pose_mappings = {

				   "LTHI": 1,
				   "LKNE": 2,
				   "LANK": 3,
				   "RTHI": 4,
				   "RKNE": 5,
				   "RANK": 6,

}

detectron_to_openpose = {
	0:0,
	1:0, 
	2:6,
	3:8,
	4:10,
	5:5,
	6:7,
	7:9,
	8:12,
	9:14,
	10:16,
	11:11,
	12:13,
	13:15,


}

def convert_detectron_to_open_pose(detectron_keypoints):
	"""
		Converts array indexes ID's from detectron to openpose to prepare for 3d pose lifting

		Parameters:
			detectron_keypoints (np.array): shape => (joint ID, dim) (3rd dim is a score not Z)
		Returns:
			(np.array): transformed 2d_pose into openpose format

	"""
	detectron_keypoints = detectron_keypoints[0] if len(np.shape(detectron_keypoints)) == 3 else detectron_keypoints
	res = [None for _ in range(14)]
	for i in range(14):
		res[i] = detectron_keypoints[detectron_to_openpose[i]][:-1]

	res = np.asarray(res)
	return res




class Processor:
	"""
		Holds the options and pose processing methods for videos.

		Fields:
			w: height of video
			h: width of video
			pose_estimator: pose estimator to use when processing
			poseLifting: 3d pose lifting model to use when processing
	"""
	def __init__(self, args):
		self.w, self.h = model_wh(args.resize)

		if self.w == 0 or self.h == 0:
			self.pose_estimator = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
		else:
			self.pose_estimator = TfPoseEstimator(get_graph_path(args.model), target_size=(self.w, self.h))
		self.poseLifting = Prob3dPose('lifting/models/prob_model_params.mat')

	def process_video(self,video_file,target_filename,one_person = True):

		"""
			Gets relevant 3d_poses from raw video via openpose and output a pose visualization video.

			Parameters:
				video_file (str): Video filename of target video
				target_filename (str): location of where to store processed video
				one_person (bool): If true only detects the pose of the largest person on the first frame of the video.
					If false will get all poses of people in video.

			Returns:
				(list) 3d_pose or poses from people in specificed video

		"""
		cap = cv2.VideoCapture(video_file)
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		width = cap.get(3)
		height= cap.get(4)
		out = cv2.VideoWriter(target_filename,fourcc,25.0,(int(width),int(height)))
		if(not cap.isOpened()):
			raise Exception("Error Opening Video")

		midpoint = None
		vicon_data = []
		while(cap.isOpened()):
			ret,frame = cap.read()
			if ret:
			    t = time.time()

			    humans = self.pose_estimator.inference(frame,resize_to_default=(self.w > 0 and self.h > 0), upsample_size = 4.0)
			    if one_person:
				    closest_person_index,midpoint = Processor.grab_closest_human(humans,frame,last_closest = midpoint)
				    if closest_person_index == -1:
				    	print('Rogers frame not detected')
				    	Processor.draw_poses(humans,frame)
				    	vicon_data.append(np.zeros((3,17)))
				    else:
				    	#Processor.draw_poses([humans[closest_person_index]],frame)
				    	cv2.circle(frame, midpoint, 3, (0,0,0), thickness=3, lineType=8, shift=0)
				    	
				    	vicon_data.append(self.grab_keypoints([humans[closest_person_index]],frame.shape[:2],frame)[0])
				    	
				    	#vicon_data.append(np.zeros((3,17)))

			    
			    #plot_pose(pose_3d[closest_person_index])
			    
			    elapsed = time.time() - t
			    #logger.info('inference image in %.4f seconds.' % (elapsed))
			    #Processor.draw_poses(humans,frame)
			    cv2.imshow('Frame',frame)
			    out.write(frame)
			    key = cv2.waitKey(25)
			    if key == ord('q'):
			    	break
			else:
				break
		
		cap.release()
		out.release()

		return np.array(vicon_data)


	def grab_keypoints(self,humans,image_dimensions,image):
		"""
			Returns 3d_poses from 2d_poses stored in human objects and draws centered pose onto specified frame.

			Parameters:
				humans (list: list of Human objects that contain 2d_pose information
				image_dimensions (tuple): (w,h) of frame
				image (np.array): The relevant image that the humans object was extracted from

			Returns:
				(list): 3d_poses of humans

		"""
		
		image_h, image_w = image_dimensions
		pose_2d_mpiis = []
		visibilities = []
		for human in humans:
			pose_2d_mpii,visibility = common.MPIIPart.from_coco(human)
			centered_pose_2d_mpii = [(int(x * image_w + .5),int(y* image_h + .5)) for x,y in pose_2d_mpii]
			pose_2d_mpiis.append(pose_2d_mpii)
			Processor.draw_detectron_poses(centered_pose_2d_mpii,image)
			visibilities.append(visibility)
		pose_2d_mpiis = np.array(pose_2d_mpiis)
		
		visibilities = np.array(visibilities)
		transformed_pose2d,weights = self.poseLifting.transform_joints(pose_2d_mpiis,visibilities)
		pose_3d = self.poseLifting.compute_3d(transformed_pose2d,weights)

		#plot_pose(pose_3d[0])
		#plt.show()

		return pose_3d



	@staticmethod
	def draw_poses(humans, frame):
		"""
			Draws human poses onto frame from openpose data.

			Parameters:
				humans (list): List of Human objects that contain 2d_pose information
				frame (np.array): Image to draw poses on
			Returns:
				None

		"""
		image_h, image_w = frame.shape[:2]
		centers = {}
		for human in humans:
			for i in range(common.CocoPart.Background.value):
				if i not in human.body_parts.keys():
					continue
		
				body_part = human.body_parts[i]
				center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
				centers[i] = center
				cv2.circle(frame, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)
				
			# draw line
			for pair_order, pair in enumerate(common.CocoPairsRender):
				if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
					continue
				cv2.line(frame, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)

	def draw_detectron_poses(human,frame):
		"""	

		Draws human poses onto frame from detectron data.

		Parameters:
			human (np.array): shape => (joint ID, dim) , np array of 2d pose
			frame: Image to draw human poses on

		Returns:
			None


		"""
		image_h, image_w = frame.shape[:2]
		for i,keypoint in enumerate(human):
			#center = (int(keypoint[0] * image_w + 0.5), int(keypoint[1] * image_h + 0.5))
			#cv2.circle(frame, tuple(keypoint), 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)
			cv2.putText(frame, str(i),tuple(keypoint),cv2.FONT_HERSHEY_SIMPLEX,.35,3)



	@staticmethod
	def grab_closest_human(humans,frame,last_closest = None):

		"""

			Gets the index and pose thats closest to the point specified. If the point specified is None, it will return the largest.

			Parameters:
				humans (list): list of human objects containing 2d_poses
				frame: Relevant image that humans are extracted from
				last_closest: reference point to determine closest pose

			Returns:
				(int): index in humans where the closest pose occurs
				(np.array): closest pose

		"""


		thresh = 90
		image_h, image_w = frame.shape[:2]
		mins = []
		midpoints = []
		for human in humans:
			y_min = min([human.body_parts[i].y for i in human.body_parts.keys()])
			y_max = max([human.body_parts[i].y for i in human.body_parts.keys()])
			x_min = min([human.body_parts[i].x for i in human.body_parts.keys()])
			x_max = max([human.body_parts[i].x for i in human.body_parts.keys()])
			#cv2.rectangle(frame,(int(x_min * image_w + .5),int(y_min * image_h +.5)),(int(x_max * image_w + .5),int(y_max * image_h + .5)),(0,0,0),3)
			midpoints.append(((int(x_max * image_w + .5) + int(x_min * image_w + .5)) // 2, (int(y_max * image_h + .5) + int(y_min * image_h + .5)) // 2))
			mins.append(-y_max)


		identifiers = [distance.euclidean(last_closest,midpoint) for midpoint in midpoints] if last_closest else mins
		try:
			human_index = np.argmin(identifiers)
		except:
			return -1,last_closest
	
		midpoint = midpoints[human_index]

		#Not correct ID
		if last_closest and distance.euclidean(midpoint,last_closest) > thresh:
			cv2.circle(frame, midpoint, 3, (0,0,0), thickness=3, lineType=8, shift=0)
			return -1, last_closest

		return human_index , midpoints[human_index]


	@staticmethod
	def combine_videos(video_files,target_filename):
		"""
			Combines video files and stores into output path.

			Parameters:
				video_files (list): List of video file paths to combine
				target_filename (str): Location to store combined video.

			Returns:
				None

		"""

		if not video_files[0]:
			raise Exception("Argument video_files is empty.")

		cap = cv2.VideoCapture(video_files[0])
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		width = cap.get(3)
		height= cap.get(4)
		out = cv2.VideoWriter(target_filename,fourcc,25.0,(int(width),int(height)))

		for file in video_files:
			cap = cv2.VideoCapture(file)

			if(not cap.isOpened()):
				raise Exception("Error Opening Video in videofile {}.".format(file))

			while(cap.isOpened()):
				ret,frame = cap.read()
				if ret:
				    t = time.time()
				    elapsed = time.time() - t
				    logger.info('inference image in %.4f seconds.' % (elapsed))
				    out.write(frame)
				    key = cv2.waitKey(25)
				    if key == ord('q'):
				    	break
				else:
					break
			
			cap.release()
		out.release()



	def visualize_detectron_keypoints(self,video_file,target_filename,detectron_keypoints):

		"""
			Stores a video with 2d pose overlayed on top of original video and returns 3d_poses from detectron 2d keypoints.

			Parameters:
				video_file (str): filename of target video
				target_filename (str): Location of where to store overlayed video
				detectron_keypoints (np.array): 2d keypoints from detectron2

			Returns:
				(np.array): 3d poses of shape => (time,joint ID,dim)

		"""
		detectron_keypoints = np.array([convert_detectron_to_open_pose(x) for x in detectron_keypoints])
		cap = cv2.VideoCapture(video_file)
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		width = cap.get(3)
		height= cap.get(4)
		out = cv2.VideoWriter(target_filename,fourcc,25.0,(int(width),int(height)))
		if(not cap.isOpened()):
			raise Exception("Error Opening Video")

		frameNo = 0
		pose_3d_data = []
		while(cap.isOpened()):
			ret,frame = cap.read()
			if ret:
				human = detectron_keypoints[frameNo]
				Processor.draw_detectron_poses(human,frame)
				visibility =  [True if i != 1 else False for i in range(14)]
				transformed_pose2d,weights = self.poseLifting.transform_joints(np.array([human]),np.array([visibility]))
				pose_3d = self.poseLifting.compute_3d(transformed_pose2d,weights)
				pose_3d_data.append(pose_3d[0])
				plot_pose(pose_3d[0])
				plt.show()
				cv2.imshow('Frame',frame)
				out.write(frame)
				key = cv2.waitKey(25)
				if key == ord('q'):
					break
				frameNo +=1
			else:
				break
		
		cap.release()
		out.release()
		return pose_3d_data


	def pose_3d_from_detectron_npy(self,filename):
		"""
		Note: Same as visualize_detectron_keypoints() but without the visualization
		returns 3d_poses from detectron 2d keypoints.

		Parameters:
			filename (str): npy file location of 2d keypoints from detectron2

		Returns:
			(np.array): 3d poses of shape => (time,joint ID,dim)

		
		"""
		keypoints = np.load(filename,allow_pickle = True)[0]['keypoints']
		keypoints = np.array([convert_detectron_to_open_pose(x) for x in keypoints])

		poses = []
		for i,pose in enumerate(keypoints):
			visibility =  [True if i != 1 else False for i in range(14)]
			transformed_pose2d,weights = self.poseLifting.transform_joints(np.array([pose]),np.array([visibility]))
			try:
				pose_3d = self.poseLifting.compute_3d(transformed_pose2d,weights)
				poses.append(pose_3d[0])
			except:
				#if pose_3d fails fill with naive interpolation of last point
				if len(poses):
					poses.append(poses[-1])

		return poses










class Vicon_Processor:
	"""
		Handles Processing 3d_pose information from both vicon data and pose estimation outputs.
	"""

	def __init__():
		pass

	@staticmethod
	def pose_graph_animation(pose_data,filename):
		"""
			Returns a gif of 3d_pose plot over time.

			Parameters:
				pose_data (list): 3d_pose over time
				filename (str): filename of target path to store gif
			Returns:
				None
		
		"""

		def get_image_from_pose(pose_3d):
			fig = plot_pose(pose_3d)
			fig.canvas.draw()
			image = np.frombuffer(fig.canvas.tostring_rgb(),dtype = 'uint8')
			image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
			return image
		imageio.mimsave(filename,[get_image_from_pose(pose_3d) for pose_3d in pose_data], fps = 5)



	@staticmethod
	def compare_vicon_and_pose_signals(pose_data,vicon_data,center_marker,target_marker):


		"""
			Plots a comparison of specified pose estimation and vicon data.

			Parameters:
				pose_data (list): 3d_poses from a subject using pose estimation
				vicon_data (list): 3d_poses from a subject using sensors
				center_marker (int): ID of the marker to center offset the target_marker from
				target_marker (int): ID of the target marker of joint t of comparison
			Returns:
				None


		"""

		vicon_data = Vicon_Processor.__format_vicon_data(vicon_data)
		print(np.shape(vicon_data))

		def center_pose_around_marker(data,marker_index):
			for i,pose in enumerate(data):
				for dim in range(3):
					data[i][dim] = data[i][dim] - data[i][dim][marker_index]

		center_pose_around_marker(pose_data,pose_mappings[center_marker])
		center_pose_around_marker(vicon_data,vicon_mappings[center_marker])

		pose_data = np.transpose(pose_data[:,:,pose_mappings[target_marker]])
		vicon_data = np.transpose(vicon_data[:200,:,vicon_mappings[target_marker]])

		
		for dim in range(2):
			#pd = [pose_data[i][dim][5] for i in range(len(pose_data))]
			plt.plot([ i / 30 for i in range(len(pose_data[dim]))],pose_data[1- dim],label = '3D Pose Estimation')
			plt.plot([i / 100 for i in range(len(vicon_data[dim]))], vicon_data[dim], color = 'red',label = 'Elderly Vicon')
			plt.ylabel('{} Position with respect to {}'.format(target_marker,center_marker))
			plt.xlabel('Time (Seconds)')
			plt.legend(loc = 'lower right')
			plt.show()
		





	@staticmethod
	def __format_vicon_data(vicon_data):
		"""
			Formats data from vicon to fit 3d_pose output format.
			
			Parameters:
				vicon_data (list): 3d pose from vicon data.
			Returns:
				(np.array): transformed data to fit pose estimation output.

		"""
		try:
			vicon_data = np.asarray(vicon_data,dtype = np.float64)
		except:
			raise Exception('Vicon Data likely has gaps in it. Shape is incompatible: {}'.format(np.shape(vicon_data)))
		print(np.shape(vicon_data))
		return  vicon_data[:,0,:,:].swapaxes(0,2)


	@staticmethod
	def load_npy_files(path):

		"""
			Loads time series data from a path of a particular subject. Format of filenames must include the year.

			Parameters:
				path (str): Location of subject folder
			Returns:
				(dict): key value pair of key = year , data = pose estimation

		"""
		filenames = os.listdir(path)

		def get_year(filename):
			if not filename[:4].isnumeric():
				raise Exception("Filename of {} is in the wrong format. The beggining of the filename must be the date in form YYYY.".format(filename))
			return int(filename[:4])

		data = {}
		for filename in filenames:
			if filename != '.DS_Store':
				data[get_year(filename)] = np.load(path + filename)

		return data

	@staticmethod
	def format_for_autoencoder(data,labels = False):
		"""
			Formats data for input into autoencoder.
			
			TODO: Figure out exactly what this data object is.

		"""
		X = []
		Y = []
		for key in data.keys():
			x = [data[key][:32,:,pose_mappings[marker]] for marker in pose_mappings.keys()]
			#Scale Data
			#print(np.shape(x))
			X.append(x)
			Y.append(key)
		X = np.asarray(X, dtype = np.float64).transpose((0,2,1,3)).reshape(len(X),32,-1)
		X = np.swapaxes(X,1,2)

		for i in range(len(X)):
			for j in range(len(X[i])):
				X[i][j] = MinMaxScaler().fit_transform(X[i][j].reshape(-1,1)).reshape(-1)

		X = np.swapaxes(X,1,2)
		res = (X,Y) if labels else X
		return res




def detectron_test(args,filename):
	"""
		Gets 3d_poses from filename of detetcron 2d_pose (npy)

		Parameters:
			args (): arguments for needed for Processor models
			filename: filename of npy file of detectron 2d poses
	"""
	processor = Processor(args)
	pose_data = processor.pose_3d_from_detectron_npy(filename)
	return pose_data
		






if __name__ == "__main__":

	

	
	parser = argparse.ArgumentParser(description='tf-pose-estimation run')
	parser.add_argument('--image', type=str, default='./images/p1.jpg')
	parser.add_argument('--model', type=str, default='mobilenet_thin',
	                    help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
	parser.add_argument('--resize', type=str, default='0x0',
	                    help='if provided, resize images before they are processed. '
	                         'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
	parser.add_argument('--resize-out-ratio', type=float, default=4.0,
	                    help='if provided, resize heatmaps before they are post-processed. default=1.0')
	args = parser.parse_args()
	detectron_test(args)
	
	'''
	processor = Processor(args)
	process_path = 'Roger Federer/Walking/Videos/'
	source_path = ''
	vicon_data = processor.process_video(process_path + '2013.mp4', 'test.mp4',one_person = True)
	'''
	

	'''


	#Processor.combine_videos([process_path + filename for filename in os.listdir(process_path) if 'one' in filename],'combined-one.mp4')




	#RIGHT HIP = 4
	#RIGHT KNEE = 5
	#RIGHT ANKLE = 6
	#LEFT HIP = 1
	#LEFT KNEE = 2
	#LEFT ANKLE = 3
	
	data = np.load('Roger Federer/Vicon/2013.npy')
	
	
	centered_data = copy.deepcopy(data)
	for i,pose in enumerate(centered_data):
		for dim in range(3):
			centered_data[i][dim] = centered_data[i][dim] - centered_data[i][dim][1]


	for i in range(len(centered_data)):
		print(centered_data[i][1][1],centered_data[i][1][5])
	


	
	filename = '../../Processed Data/EFR015Trial 07/marker.npy'
	marker_data = np.load(filename, allow_pickle = True)
	#marker_data = np.load('test.npy')
	#Vicon_Processor.pose_graph_animation(data)\

	Vicon_Processor.compare_vicon_and_pose_signals(data,marker_data,'LTHI','RKNE')
	'''
	

	


