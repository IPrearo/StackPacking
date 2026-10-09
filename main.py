import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.colors as mcolors
from matplotlib.patches import Circle

from copy import deepcopy

import yaml

epsilon = 1e-8
polar_to_cartesian = lambda r, t: np.array([r*np.cos(t), r*np.sin(t)])


class Ring:
	def __init__(self, outer_diameter, inner_diameter):
		self.outer_diameter = outer_diameter
		self.inner_diameter = inner_diameter

	@property
	def outer_diameter(self):
		if not hasattr(self, '_outer_diameter'):
			raise( ValueError("Outer diameter is not set in this object.") )
		return self._outer_diameter

	@outer_diameter.setter
	def outer_diameter(self, value):
		# if self.inner_diameter > value:
			# raise( ValueError("Outer diameter must be greater than inner diameter.") )
		self._outer_diameter = float(value)


	@property
	def inner_diameter(self):
		if not hasattr(self, '_inner_diameter'):
			return 0
		return self._inner_diameter
	@inner_diameter.setter

	def inner_diameter(self, value):
		if value < 0:
			raise( ValueError("Inner diameter must be >= 0.") )
		# if self.outer_diameter < value:
			# raise( ValueError("Inner diameter must be lesser than outer diameter") )

		self._inner_diameter = float(value)


	@property
	def ratio(self):
		return self.inner_diameter / self.outer_diameter


class Capillary(Ring):
	def __init__(self, position_x=0, position_y=0, *args, **kwargs):
		self._position = np.zeros(2)
		self.position = [position_x, position_y]
		super().__init__(*args, **kwargs)

	@property
	def position_x(self):
		return self._position[0]
	
	@position_x.setter
	def position_x(self, value):
		self._position[0] = float(value)

	@property
	def position_y(self):
		return self._position[1]
	
	@position_y.setter
	def position_y(self, value):
		self._position[1] = float(value)

	@property
	def position(self):
		return self._position
	
	@position.setter
	def position(self, value):
		if len(value) != 2:
			raise( ValueError("Position must be 2D.") )
		self.position_x = value[0]
		self.position_y = value[1]


	def as_dict(self):
		return {'position_x': self.position_x,
				'position_y': self.position_y,
				'outer_diameter': self.outer_diameter,
				'inner_diameter': self.inner_diameter}


	def to_str(self):
		ret = ''
		dict_like = self.as_dict()
		for k,v in dict_like.items():
			ret += f"{k}: {v}\t"
		return ret[:-1]


	def copy(self):
		return Capillary(**self.as_dict())


	def is_inside(self, capillary):
		'''
			Checks if either this capillary is inside the provided one or the other way around,
				returning 1 if they are nested, or 0 if they aren't.

			capillary:Capillary     Capillary to check against
		'''

		# Checks if the capillaries can be inside one another
		if (self.outer_diameter > capillary.inner_diameter and \
			capillary.outer_diameter > self.inner_diameter):
			return False

		center_distance = np.linalg.norm(self.position - capillary.position)
		if self.outer_diameter > capillary.outer_diameter:
			bigger_capillary = self
			smaller_capillary = capillary
		else:
			bigger_capillary = capillary
			smaller_capillary = self
		
		radius_diff = 0.5 * (bigger_capillary.inner_diameter - smaller_capillary.outer_diameter)
		return center_distance <= radius_diff


	def is_outside(self, capillary):
		'''
			Checks if the capillaries are outside each other,
				returning 1 if they are nested, or 0 if they aren't.

			capillary:Capillary     Capillary to check against
		'''
		radius_sum = 0.5 * (self.outer_diameter + capillary.outer_diameter)
		center_distance = np.linalg.norm(self.position - capillary.position)
		
		return center_distance > radius_sum


	def is_colliding(self, capillary):
		if self.is_outside(capillary): return False
		if self.is_inside(capillary): return False
		return True



def export_capillary_list(outside_diameter, capillary_list, text_path=None, image_path=None):
	if text_path is not None:
		with open(text_path, 'w') as fp:
			fp.write(f"outside diameter: {outside_diameter}\n")
			fp.write(f"capillary list:\n")
			for c in capillary_list:
				fp.write(c.to_str()+'\n')
	
	if image_path is None:
		return

	fig = plt.figure(figsize=(10,10))
	ax = fig.add_axes(111)
	ax.add_patch( Circle((0,0), outside_diameter/2, facecolor='grey', alpha=0.5) )

	colors = list( mcolors.TABLEAU_COLORS.values() )
	color_index = 0
	for c in capillary_list:
		color = colors[color_index]
		ax.add_patch( Circle(c.position, c.outer_diameter/2, facecolor=color) )
		color_index = (color_index+1) % len(colors)

	ax.set_xlim(-outside_diameter/2, outside_diameter/2)
	ax.set_ylim(-outside_diameter/2, outside_diameter/2)
	fig.savefig(image_path, dpi=200)


def random_circular_position(max_diameter):
	r = np.sqrt( np.random.random() ) *0.5*max_diameter
	theta = np.random.random() * 2*np.pi
	return np.array([np.cos(theta)*r, np.sin(theta)*r])


def closest_capillary(index, capillary_list):
	capillary = capillary_list[index]
	closest_dist = None
	for i, c in enumerate(capillary_list):
		if i == index: continue

		dist = np.linalg.norm(c.position - capillary.position)
		if closest_dist is None:
			closest_dist = dist
			continue
		if closest_dist > dist:
			closest_dist = dist
			continue

	if closest_dist is None:
		closest_dist = np.inf
	return closest_dist


def cost_function(outer_capillary, capillary_list):
	penalty_cost = 2000
	reward_mult = 80*len(capillary_list)
	cost = 0
	for i, c in enumerate(capillary_list):
		cost -= c.ratio*reward_mult
		if not c.is_inside(outer_capillary):
			cost+=penalty_cost
		
		for c2 in capillary_list[i+1:]:
			if c.is_inside(c2):
				cost+=penalty_cost
				continue
			if c.is_colliding(c2):
				cost+=penalty_cost
	return cost


def annealing_packing(outside_diameter, diameter_df, must_have=None,
					  temp=1e5, min_temp=1e-10, cool_rate=0.9999,
					  max_iter=100000, cooling_type='exp'):

	T_start = temp

	diameter_df['Ratio'] = diameter_df['Internal Diameter'] / diameter_df['External Diameter']
	diameter_df.sort_values(by='Ratio', ascending=False)
	N_diameters = diameter_df.shape[0]

	lesser_diameter = np.min(diameter_df['External Diameter'])
	greater_diameter = np.max(diameter_df['External Diameter'])
	position_diameter = outside_diameter - lesser_diameter

	outer_capillary = Capillary(0, 0, outside_diameter, outside_diameter)
	if must_have is None:
		capillary_list = []
	else:
		if np.ndim(must_have) > 0:
			capillary_list = must_have
		else:
			capillary_list = [must_have]

	must_have_N = len(capillary_list)
	# Absolute maximum number of inner cappilaries if they were to completelly pack the outer one
	max_N = int(outside_diameter**2 / lesser_diameter**2)
	min_N = max( int(outside_diameter**2 / greater_diameter**2), must_have_N+1 )

	def change_position(index, c_list):
		c_list[index].position += random_circular_position(position_diameter)*temp/T_start
		return c_list[index]
			
	def change_diameter(index, c_list):
		diameter_index = np.random.randint(0, N_diameters)
		c_list[index].inner_diameter = diameter_df.iloc[diameter_index]['Internal Diameter']
		c_list[index].outer_diameter = diameter_df.iloc[diameter_index]['External Diameter']
		return c_list[index]

	def new_capillary(c_list):
		# The only possible pivot capillary is the outside one
		if len(c_list) == 0:
			d_index = np.random.randint(0, N_diameters)
			df_line = diameter_df.iloc[d_index]
			angle = np.random.random()*2*np.pi
			r = 0.5*(outside_diameter - df_line['External Diameter'])
			position = polar_to_cartesian(r, angle)
			c = Capillary(position_x=position[0], position_y=position[1],
						outer_diameter=df_line['External Diameter'], inner_diameter=df_line['Internal Diameter'])
			c_list.append(c)
			return True

		good_pivots = False
		# Tries a limited number of times to find pivots that allow for new capillaries
		for i in range(20):
			if good_pivots: break

			# Chooses 2 pivot capillaries for the new one to be tangential to 
			pivot_indexes = np.random.choice(np.arange(len(c_list)+1), 2, replace=False)
			# The outer one is special as all capillaries are inside it instead of outside
			has_outside = len(c_list) in pivot_indexes
			# Ensures the outside one is on the zero index so we don't need to code both cases
			if pivot_indexes[1] == len(c_list):
				pivot_indexes = [pivot_indexes[1], pivot_indexes[0]]

			if has_outside:
				pivots = [outer_capillary, c_list[pivot_indexes[1]]]
			else:
				pivots = [c_list[i] for i in pivot_indexes]

			pivot_distance = np.linalg.norm(pivots[0].position - pivots[1].position)
			
			# Handles the case of a capillary in the middle and the outer capillary being chosen together
			if pivot_distance < epsilon:
				continue

			if has_outside:
				pivot_wall_distance = -pivot_distance + 0.5*(outside_diameter - pivots[1].outer_diameter)
			else:
				pivot_wall_distance = pivot_distance - 0.5*(pivots[0].outer_diameter + pivots[1].outer_diameter)

			if pivot_wall_distance <= greater_diameter + epsilon:
				good_pivots = True
		if not good_pivots:
			return False

		# Filters only sufficient capillary diameters
		limited_df = diameter_df[diameter_df['External Diameter'] >= pivot_wall_distance]
		N_limited_df = limited_df.shape[0]

		d_index_order = np.random.choice(np.arange(N_limited_df), N_limited_df, replace=False)
		d_index = 0
		bad_position = True
		while bad_position:
			# Fails to add a new capillary for this pivot
			if d_index >= N_limited_df: return

			df_line = diameter_df.iloc[d_index]
			out_d = df_line['External Diameter']

			max_pos_r = 0.5*(outside_diameter - out_d)

			R = [0.5*(p.outer_diameter + out_d) for p in pivots]
			if has_outside:
				R[0] = 0.5*(outside_diameter - out_d)

			# Calculates the position of intersection for two circles in the X axis
			#	with the same radii as the pivots+the chosen capillary
			x_prime2 = (pivot_distance**2 - R[1]**2 + R[0]**2) / (2*pivot_distance)
			y_prime2_squared = R[0]**2 - x_prime2**2 + epsilon

			x_prime2 = np.ones(2)*x_prime2
			try:
				y_prime2 = np.array([ np.sqrt(y_prime2_squared), -np.sqrt(y_prime2_squared) ], dtype=float)
			except(e):
				print(f"y_prime2_squared: {y_prime2_squared}\nx_prime2: {x_prime2}")
				y_prime2 = np.array([ np.sqrt(y_prime2_squared), -np.sqrt(y_prime2_squared) ], dtype=float)

			positions_prime2 = [x_prime2, y_prime2]

			del y_prime2_squared

			# Rotates the axis such that pivot 0 is still the origin, but pivot 1 has the correct angle with pivot 0
			pos1_prime = pivots[1].position - pivots[0].position
			c = pos1_prime[0] / np.linalg.norm(pos1_prime)
			angle_prime = np.arccos( c )
			s = np.sin(angle_prime)
			rot_matrix = [[c, -s], [s, c]]
			positions_prime = np.matmul(rot_matrix, positions_prime2)

			del positions_prime2, rot_matrix, angle_prime, c, s, pos1_prime

			# Adds the pivot 0 position back, so that the calculated positions
			# 	are now in the main frame of reference
			positions = np.transpose(positions_prime) + np.array([pivots[0].position, pivots[0].position])

			del positions_prime

			# Checks if they are usable (inside the main capillary)
			usable_positions = positions[np.linalg.norm(positions, axis=1) <= max_pos_r+epsilon]
			if len(usable_positions) < 1:
				# print("positions:")
				# print(positions)
				# print(np.linalg.norm(positions, axis=1))
				# print(out_d)
				d_index += 1
				continue
			
			# Chooses one of the possible positions at random
			position = usable_positions[np.random.randint(len(usable_positions))]
			bad_position = False

		c = Capillary(position_x=position[0], position_y=position[1],
					  outer_diameter=df_line['External Diameter'], inner_diameter=df_line['Internal Diameter'])
		c_list.append(c)

		return True
		

	best_solution = capillary_list
	best_cost = cost_function(outer_capillary, capillary_list)
	last_changed_iter = 0
	cost = best_cost
	for iteration in range(max_iter):
		temp_list = deepcopy(capillary_list)
		if iteration%100 == 0:
			print(f"Iteration #{iteration}/{max_iter}.\tCurrent cost: {cost:5.0f}.\tCurrent best cost:{best_cost:5.0f}.\tTemperature={temp:.2e}")
		p = np.random.random()
		if (p < 0.25 or len(temp_list) < min_N) and (len(temp_list) < max_N):
			prev_len = len(temp_list)
			appended = new_capillary(temp_list)
			# if appended:
			# 	print("New capillary!")
		elif p < 0.5 and len(temp_list) > must_have_N:
			index = np.random.randint(must_have_N, len(temp_list))
			temp_list.pop(index)
		elif p < 0.75:
			index = np.random.randint(must_have_N, len(temp_list))
			temp_list[index] = change_position(index, temp_list)
		else:
			index = np.random.randint(must_have_N, len(temp_list))
			temp_list[index] = change_diameter(index, temp_list)


		new_cost = cost_function(outer_capillary, temp_list)
		delta = new_cost - cost
		if delta < 0:
			accept = True
		else:
			accept = np.random.random() < np.exp(-delta/temp)

		if new_cost < best_cost:
			best_cost = new_cost
			best_solution = deepcopy(temp_list)

		if accept:
			cost = new_cost
			capillary_list = deepcopy(temp_list)
			last_changed_iter = iteration

		# Raises temperature if it "stabilized" too much
		if iteration-last_changed_iter > 100:
			temp = max(temp, T_start * (1-iteration/max_iter))

		if cooling_type=='exp':
			temp *= cool_rate
		elif cooling_type=='lin':
			temp -= (T_start-min_temp)/max_iter
		else:
			raise(ValueError("Incorret cooling type."))
		if temp < min_temp:
			print("Stopped from temperature.")
			break

	return best_solution


def read_input(input_file):
	# This regex matches floating point numbers, including:
	#	0.2 - .2 - 2 - 200 - etc
	# float_regex =re.compile(r'[0-9]*\.?[0-9]+')

	header = 0
	with open(input_file, 'r') as fp:
		yaml_str = ""
		line = fp.readline()
		while line:
			if "=-"*5 in line: break
			yaml_str += line + '\n'
			header += 1
			line = fp.readline()

		try:
			yaml_data = yaml.safe_load(yaml_str)
		except:
			yaml_data = None

	if yaml_data is None:
		has_outer_diam = False
		required = []
	else:
		header += 1
		has_outer_diam = "outside_diameter" in yaml_data.keys()
		if "required" in yaml_data.keys():
			required = [Capillary(**r) for r in yaml_data['required']]
		else:
			required = []

	if has_outer_diam:
		outside_diameter = yaml_data['outside_diameter']
	else:
		outside_diameter = float(input("Insert outside diameter: "))

	capillary_data = pd.read_csv(input_file, sep='\t', header=header)

	return {'outside_diameter': outside_diameter,
			'required_capillaries': required,
			'capillary_data': capillary_data}
	

if __name__ == "__main__":
	from os import path, listdir, remove, walk
	from time import time

	t0 = time()
	log_path = './main.log'

	def log(message, tab_count=1):
		t1 = time() - t0
		logstr = f"{t1:.4f}s passed. {'\t'*tab_count}{message}\n"
		print(logstr, end='')
		with open(log_path, 'a+') as fp:
			fp.write(logstr)

	t_start = t0
	overall_log_path = log_path
	def main_log(*args, **kwargs):
		global t0, log_path
		t_prev = t0
		path_prev = log_path

		t0 = t_start
		log_path = overall_log_path
		log(*args, **kwargs)

		t0 = t_prev
		log_path = path_prev



	examples_path = "./Examples"
	# examples_directories = listdir(examples_path)
	examples_directories = next(walk(examples_path))[1]

	for ex_dir in examples_directories:
		main_log(f"Starting example {ex_dir}.\n")

		t0 = time()
		ex_path = path.join(examples_path, ex_dir)
		log_path = path.join(ex_path, "output.log")
		
		if path.isfile(log_path):
			remove(log_path)
		log("Log file created.\n")

		ex_input = path.join(ex_path, "input")
		ex_txtoutput = path.join(ex_path, "output")
		ex_pngoutput = path.join(ex_path, "output.png")

		input_data = read_input(ex_input)
		outside_diameter = input_data['outside_diameter']
		required_capillaries = input_data['required_capillaries']
		capillary_data = input_data['capillary_data']

		log(f"Read capillary diameter file.")
		log(capillary_data.columns.values, tab_count=2)
		for i in range(capillary_data.shape[0]):
			ending = '\n' if i==capillary_data.shape[0]-1 else ''
			log(str(capillary_data.iloc[i].values) + ending, tab_count=2)

		log(f"Required capillaries:")
		for i, req_c in enumerate(required_capillaries):
			ending = '\n' if i==len(required_capillaries)-1 else ''
			log(req_c.to_str() + ending, tab_count=2)
			
		log(f"Packing for outside diameter of {outside_diameter:.4f}.\n")

		log("Starting annealing packing.\n")
		optimized = annealing_packing(outside_diameter, capillary_data, must_have=required_capillaries)
		log("Finished annealing packing.\n")
		log("Exporting annealing packing.\n")
		export_capillary_list(outside_diameter, optimized, ex_txtoutput, ex_pngoutput)

		main_log(f"Finished example {ex_dir}.\n")


	main_log(f"Finished program.")