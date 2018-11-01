import numpy as np
import pandas as pd
def find_distance(body_part_1, body_part_2, method = 'euclidean'):
	"""
	Computes the distances between any two body parts, given x, y coordinates
	INPUT 
	body_part_1  | [x, y] array of the position of body_parts
	"""
	if method == 'euclidean':
		dist = numpy.linalg.norm(body_part_1 - body_part_2)

def main():
	"""
	Computes distance over all body parts
	"""
	body_part_list = ['nose', 'tail']

	