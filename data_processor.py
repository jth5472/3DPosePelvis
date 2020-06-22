import math
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
from scipy.stats import ttest_ind,mannwhitneyu
import argparse
from video_processor import detectron_test
import os
import matplotlib.patches as mpatches


folder_path = 'time_data/3d_poses/'



INT_TO_DIM = {
	0: 'X',
	1: 'Y',
	2: 'Z'
}


class Subject:
	def __init__(self,id,parkinsons,pose_3d,date = -1,group = -1): 
		self.uuid = id
		self.parkinsons = parkinsons
		self.pose_3d = pose_3d
		self.date = date
		self.group = group

	def process_features(self,fp):
		features = {}
		pelvis_angles = fp.pelvis_angles(self.pose_3d)
		pelvis_accelerations = fp.pelvis_accelerations(self.pose_3d)
		features["pelvis_angle_diff"] = pelvis_angles['diff']
		features["accelerations_diff"] = pelvis_accelerations['diff']
		#plot_all_dims(pelvis_accelerations['time_series'],"Pelvis Angle Pos {} , Subject: {}".format("Parkinsons" if self.parkinsons else "Non-Parkinsons",self.uuid), 'blue' if self.parkinsons else 'red')
		return features

	def __repr__(self):
		return 'ID:{},parkinsons: {}, date: {} , group: {}' 'sample_length: {}'.format(self.uuid,self.parkinsons,self.date,self.group,len(self.pose_3d))




class FeatureProcessor:
	def __init__(self):
		pass




	def pelvis_angles(self,pose_3d):
		'''
		@params: pose_3d data of shape => (frames,dimensions,joint ID)
		'''
		pelvis_angles = {'time_series': [[] for _ in range(3)] , 'diff': [-1 for _ in range(3)]}
		pelvis_joint_index = 0


		for pose in pose_3d:
			for dim in range(3):
				pelvis_point = [pose[0][pelvis_joint_index],pose[1][pelvis_joint_index],pose[2][pelvis_joint_index]]
				pelvis_point[dim] = 0
				pelvis_point = np.array(pelvis_point)

				''' any point along the targeted plane axis '''
				calculate_point = [0,0,0]
				calculate_point[(dim + 1) % 3] = 5
				calculate_point = np.array(calculate_point)

	

				def mag(x):
					return np.sqrt(x.dot(x))

				joint_angle = math.acos(np.dot(pelvis_point,calculate_point) / (mag(pelvis_point) * mag(calculate_point)))	
				joint_angle = (joint_angle * 180 /(math.pi)) % 360
				pelvis_angles['time_series'][dim].append(joint_angle)
		
		for dim in range(3):
			pelvis_angles['diff'][dim] = max(pelvis_angles['time_series'][dim]) - min(pelvis_angles['time_series'][dim])


		
		return pelvis_angles


	def pelvis_accelerations(self,pose_3d):
		pelvis_acceleration = {'time_series': [[None] for _ in range(3)] , 'diff': [-1 for _ in range(3)]}
		for dim in range(3):
			pelvis_positions =  np.asarray([x[dim][0] for x in pose_3d])
			pelvis_accelerations = np.diff(np.diff(pelvis_positions))
			pelvis_acceleration['time_series'][dim] = pelvis_accelerations
			pelvis_acceleration['diff'][dim] = max(pelvis_accelerations) - min(pelvis_accelerations)

		return pelvis_acceleration
		


def get_subjects(split_subjects = False, frame_step = 35):
	parkinsons_string = 'parkinsons'
	non_parkinsons_string = 'non_parkinsons'
	folder_paths = ['processed_data/{}/'.format(non_parkinsons_string),'processed_data/{}/'.format(parkinsons_string)]
	subjects = {parkinsons_string: {}, non_parkinsons_string: {}}
	for folder in folder_paths:
		for filename in os.listdir(folder):
			if filename.split('.')[1] == 'npy':
				parkinsons_identifier,uuid = filename.split('.')[0].split('-')
				uuid = int(uuid)
				parkinsons = True if parkinsons_identifier == parkinsons_string else False

		

				if split_subjects:
					pose_3d = np.load(folder + filename)
					for i in range(frame_step,len(pose_3d),frame_step):
						trial_uuid = int("{}{}".format(uuid,i // frame_step))
						subjects[parkinsons_identifier][trial_uuid] = Subject(trial_uuid,parkinsons,pose_3d[i - frame_step:i],group = uuid)

				else:

					subjects[parkinsons_identifier][uuid] = Subject(uuid,parkinsons,np.load(folder + filename))
					
	return subjects





def get_time_subjects(frame_step = 30):
	subjects = {}
	for i,filename in enumerate(os.listdir(folder_path)):
		if filename != ".DS_Store":
			parkinsons = True if filename[0] == 'p' else False
			
			date_str = filename.split('.')[0]
			date_str = date_str if not parkinsons else date_str[1:]
			(m,d,y) = date_str.split(':')
			if not parkinsons and int(y) < 2000:
				continue
			current_date = datetime.date.today()
			filename_date = datetime.date(int(y),int(m),int(d))
			delta = (filename_date - current_date).days
			pose_3d = np.load(folder_path + filename)
			for j in range(frame_step,len(pose_3d),frame_step):
				uuid = '{} {}'.format(i,j)
				subjects[uuid] = Subject(uuid,parkinsons,pose_3d[j - frame_step:j],date = int(y))
	return subjects

def graph_time_series(feature_data,avg = False):
	plot_data = {'parkinsons': 

				{'Z_accelerations': [],
				 'X_accelerations': [],
				 'Y_accelerations':  [],
				 'Z_pelvis': [],
				 'Y_pelvis': [],
				 'X_pelvis': [],
				 },

				 'non_parkinsons': 

				{'Z_accelerations': [],
				 'X_accelerations': [],
				 'Y_accelerations':  [],
				 'Z_pelvis': [],
				 'Y_pelvis': [],
				 'X_pelvis': [],
				 },

	}

	for class_key in feature_data:
		for feature_key in feature_data[class_key]:
			time = feature_data[class_key][feature_key]['date']
			plot_data[class_key]['Z_accelerations'].append((feature_data[class_key][feature_key]['accelerations_diff'][2],time))
			plot_data[class_key]['X_accelerations'].append((feature_data[class_key][feature_key]['accelerations_diff'][0],time))
			plot_data[class_key]['Y_accelerations'].append((feature_data[class_key][feature_key]['accelerations_diff'][1],time))
			plot_data[class_key]['Z_pelvis'].append((feature_data[class_key][feature_key]['pelvis_angle_diff'][2],time))
			plot_data[class_key]['Y_pelvis'].append((feature_data[class_key][feature_key]['pelvis_angle_diff'][1],time))
			plot_data[class_key]['X_pelvis'].append((feature_data[class_key][feature_key]['pelvis_angle_diff'][0],time))


	def transform_data(data):
		grouped_by_subject = {}
		for (feature,time) in data:
			grouped_by_subject[time] = grouped_by_subject.get(time,[])
			grouped_by_subject[time].append(feature)

		x,y = [],[]
		for (time, points) in grouped_by_subject.items():
			x.append(time)
			if avg:
				y.append(np.mean(points))
			else:
				y.append(np.median(points))
		return x,y


	for key in plot_data['parkinsons']:

			for class_key in plot_data:


				(x,y) = transform_data(plot_data[class_key][key])
				
			
	
				plt.scatter(x,y, color = 'red' if class_key == 'parkinsons' else 'blue', s = 70)

				if 'accelerations' not in key:
					plt.ylabel("Degrees",fontsize = 21)
				else:
					plt.ylabel("Acceleration",fontsize = 21)
				if class_key == "parkinsons":
					plt.title('Muhammad Ali', fontsize = 35)
				else:
					plt.title("Vanna White",fontsize = 35)

				plt.xlabel("Time",fontsize = 18)
				plt.yticks(fontSize = 15)
				plt.xticks(fontSize = 15)
				plt.locator_params(nbins=6)
				plt.savefig("time_series/{}_{}.png".format(class_key,key))
				plt.show()
				


def get_time_features(subjects):
 fp = FeatureProcessor()
 features = {'parkinsons': {}, 'non_parkinsons': {},}
 for subject_key in subjects:
 	subject = subjects[subject_key]
 	feature = subject.process_features(fp)	
 	feature['date'] = subject.date
 	features['parkinsons' if subject.parkinsons else 'non_parkinsons'][subject.uuid] = feature
 return features


def show_seperation(feature_data):


	for feature_key in feature_data:
		features = feature_data[feature_key]
		print("{}:".format(feature_key))
		for n in range(3):
	

			fig,axes = plt.subplots(2)
			fig.suptitle('{} Dim:{} - park(blue) vs non-park(red)'.format(feature_key,INT_TO_DIM[n]))

			axes[0].eventplot(features['parkinsons'][n], orientation = 'horizontal', linelengths = .5, color = 'blue')
			axes[0].get_yaxis().set_visible(False)
			axes[1].eventplot(features['non_parkinsons'][n], orientation = 'horizontal', color = 'red')
			axes[1].get_yaxis().set_visible(False)

			min_val = min(features['parkinsons'][n] + features['non_parkinsons'][n])
			max_val = max(features['parkinsons'][n] + features['non_parkinsons'][n])
			axes[0].set_xlim([min_val,max_val])
			axes[1].set_xlim([min_val,max_val])
			plt.xlabel("Pelvis Accelerations Peak Cycle Diff (pixel/frame^2)" if feature_key == "pelvis_accelerations" else "Pelvis Angles Peak Cycle Diff (Degrees)")
			plt.show()

def t_tests(feature_data,alpha = .05):
	for feature_key in feature_data:
			features = feature_data[feature_key]
			for n in range(3):
				stat, p = mannwhitneyu(features['parkinsons'][n],features['non_parkinsons'][n])
				print('For Param {} Dim {}: Statistics= {}, p= {}'.format(feature_key,n,stat, p))
				# interpret
				if p > alpha:
					print('Same distributions (fail to reject H0)')
				else:
					print('Different distributions (reject H0)')
				print()



def show_histograms(feature_data):
	for feature_key in feature_data:
		features = feature_data[feature_key]
		print("{}:".format(feature_key))
		for n in range(3):
	
			fig,axes = plt.subplots(2)
			fig.suptitle('{} Dim:{} - park(blue) vs non-park(red)'.format(feature_key,INT_TO_DIM[n]))
			axes[0].hist(features['parkinsons'][n], color = 'blue')
			axes[1].hist(features['non_parkinsons'][n], color = 'red')
			min_val = min(features['parkinsons'][n] + features['non_parkinsons'][n])
			max_val = max(features['parkinsons'][n] + features['non_parkinsons'][n])
			plt.xlabel("Pelvis Accelerations Peak Cycle Diff (pixel/frame^2)" if feature_key == "pelvis_accelerations" else "Pelvis Angles Peak Cycle Diff (Degrees)")
			plt.show()


def get_features():
	subjects = get_subjects(split_subjects = True)
	pelvis_accelerations,pelvis_angles = {'parkinsons': [[] for _ in range(3)], 'non_parkinsons': [[] for _ in range(3)],} , {'parkinsons': [[] for _ in range(3)], 'non_parkinsons': [[] for _ in range(3)],} 
	fp = FeatureProcessor()
	for classname in subjects:
		for subject_key in subjects[classname]:
			subject = subjects[classname][subject_key]
			parkinsons_identifier = 'parkinsons' if subject.parkinsons else 'non_parkinsons'
			features = subject.process_features(fp)
			for n in range(3):
				pelvis_accelerations[parkinsons_identifier][n].append(features["accelerations_diff"][n])
				pelvis_angles[parkinsons_identifier][n].append(features["pelvis_angle_diff"][n])

	return {
		"pelvis_accelerations": pelvis_accelerations,
		"pelvis_angles": pelvis_angles,
	}

def grouped_bar_plot(feature_data,title_key,avg = True,barWidth = .25):
	plot_data = {}
	for class_key in feature_data:
		if avg:
			plot_data[class_key] = [np.mean(feature_data[class_key][n]) for n in range(3)]
		else:
			plot_data[class_key] = [np.median(feature_data[class_key][n]) for n in range(3)]

	r = np.arange(3)
	for i,key in enumerate(plot_data):
		color = 'red' if i == 0 else 'blue'
		plt.bar(r,plot_data[key],color = color, width = barWidth,label = key)
		r = [x + barWidth for x in r]
		
	avg_string  = "(mean)" if avg else "(median)"
	title = "Acceleration " + avg_string if title_key == "pelvis_accelerations" else "Angle " + avg_string
	
	if title_key == "pelvis_angles":
		plt.ylabel("Degrees",fontsize = 25)
	plt.title(title,fontsize = 25)
	plt.xticks([x + barWidth for x in range(3)],["X","Y","Z"],fontSize = 15 )
	plt.yticks(fontSize = 15)
	plt.legend(prop =  {'size': 20,})
	plt.show()



	

def get_3d_poses(folder,output_folder):

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

	for filename in os.listdir(folder):
		if filename != ".DS_Store":
			print('hi',filename)
			pose_3d = detectron_test(args,folder + filename)
			np.save(output_folder + filename, pose_3d)


def check_gap_amount(folder):
	for filename in os.listdir(folder):
		if filename!= ".DS_Store":
			pose_2d = np.load(folder + filename,allow_pickle = True)[0]
			gap_counter = 0
			for pose in pose_2d['keypoints']:
				if np.shape(pose) == (17,3):
					gap_counter+=1
				elif sum(pose[0][0]) == 0:
					print(pose)
			print('{}:{}'.format(filename,gap_counter))


	



def main():
	subjects = get_time_subjects()
	features = get_time_features(subjects)
	graph_time_series(features)
	#grouped_bar_plot(features['pelvis_accelerations'],'pelvis_accelerations', avg = True)
	#check_gap_amount('time_data/2d_poses/')
	
	#get_3d_poses('time_data/2d_poses/','time_data/3d_poses/')

	
	



def plot_all_dims(data,title,color):
	fig, axes = plt.subplots(3)
	fig.suptitle(title)
	for i,dim_data in enumerate(data):
		axes[i].plot(dim_data,color = color)
	plt.show()


if __name__ == "__main__":
	main()
	
	
	









