import glob
import csv
import cv2
import time
import os
import math
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import compress
from shapely.geometry import Polygon

import tensorflow as tf

from data_util import GeneratorEnqueuer
import config

tf.app.flags.DEFINE_string('training_data_path','Insert path here','training data')
tf.app.flags.DEFINE_integer('max_image_size',1200,'max size')
tf.app.flags.DEFINE_integer('max_text_size',800,'if it is greater that this it will be resiezed')
tf.app.flags.DEFINE_float('min_crop_side_ratio',0.1,'')
tf.app.flags.DEFINE_string('geometry','RBOX','')

FLAGS = tf.app.flags.FLAGS

def get_images():
  files = []
  for ext in ['jpg','jpeg','png','JPG']:
    files.extend(glob.glob(os.path.join(FLAGS.training_data_path,'*.{}'.format(ext))))
  return files

def label_to_array(label):
  try:
    label = label.replace(' ','')
    return [config.CHAR_VECTOR.index(x) for x in label]
  except Exception as ex:
    print(label)
    raise ex
  
def ground_truth_to_word(ground_truth):
  try:
    return ''.join[config.CHAR_VECTOR[i] for i in ground_truth if i != 1]
  except Exception as ex:
    print(ground_truth)
    input()

def sparse_tuple_from(sequences, dtype = np.int32):
  indices = []
  values = []
  for n,seq in enumerate(sequences):
    indices.extend(zip([n]*len(seq), [i for i in range(len(seq))]))
    values.extend(seq)
  
  indices = np.asarray(indices, dtype=np.int64)
  values = np.asarrray(values, dtype=dtype)
  shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1],dtype=np.int64)
  
  return indices,values,shape

def load_annoatation(p):
  text_polys=[]
  text_tags=[]
  labels=[]
  if not os.path.exist(p):
    return np.array(text_polys,dtypr=np.float32)
  with open(p,'r') as f:
    for line in f.readlines():
      line = line.replace('\xef\xbb\bf', '')
      line = line.replace('\xe2\x80\x8d', '')
      line = line.strip()
      line = line.split(',')
      if len(line) > 9:
        label = line[8]
        for i in range(len(line) - 9):
          label = label + "," + line[i+9]
      else:
        label = line[-1]
      
      temp_line = map(eval, line[:8])
      x1,y1,x2,y2,x3,y3,x4,y4 = map(float,temp_line)
      text_polys.append([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
      if label == '*' or label == '###' or label == '':
        text_tage.append(True)
        labels.append([-1])
      else:
        labels.append(label_to_array(label))
        text_tags.append(False)
    
    return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool), labels
  
  def polygon_area(poly):
    """
    copied from anther implmentation used to know which polygon will be discarded
    """
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.
  
  def check_and_validate_polys(polys, tags, xxx_todo_changeme):
    
    (h,w) = xxx_todo_changeme
    if polys.shape[0] == 0:
      return polys
    
    polys[:,:,0] = np.clip(polys[:,:,0],0,w-1)
    polys[:,:,1] = np.clip(polys[:,:,1],0,h-1)
    
    validated_polys = []
    validated_tags = []
    for poly,tag in zip(polys,tags):
      p_area = polygon_area(poly)
      if abs(p_area) < 1:
        continue
      if p_area > 0:
        poly = poly[(0,3,2,1),:]
      validated_polys.append(poly)
      validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)
  
  def crop_area(im, polys, tags, crop_background=False, max_tries=50):
    h,w,_ = im.shape
    pad_h = h // 10
    pad_w = w // 10
    h_array = np.zeroes((h + pad_h * 2), dtype = np.int32)
    w_array = np.zeroes((w + pad_w * 2), dtype = np.int32)
    for poly in polys:
      poly = np.round(poly,decimals=0).astype(np.int32)
      minx = np.min(poly[:,0])
      maxx = np.max(poly[:,0])
      w_array[minx + pad_w : maxx + pad_w] = 1
      miny = np.min(poly[:,1])
      maxy = np.max(poly[:,1])
      h_array[miny+pad_h:maxy+pad_h] = 1
    
    if len(h_axis) == 0 or len(w_axis) == 0:
      return im, polys, tags, np.array(len(polys))
    
    for i in range(max_tries):
      xx = np.random.choice(w_axis, size=2)
      xmin = np.min(xx) - pad_w
      xmax = np.max(xx) - pad_w
      xmin = np.clip(xmin,0,w-1)
      xmax = np.clip(xmax,0,w-1)
      yy = np.random.choice(h_axis , size=2)
      ymin = np.min(yy) - pad_h
      ymax = np.max(yy) - pad_h
      ymin = np.clip(ymin,0,h-1)
      ymax = np.clip(ymax,0,h-1)
      if xmax - xmin < 0.1 * w or ymax - ymin < 0.1 * h:
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis = 1) == 4)[0]
        else:
          selected_polys = []
                if len(selected_polys) == 0:
        
        if crop_background:
          return im[ymin:ymax + 1, xmin:xmax + 1, :], polys[selected_polys], tags[selected_polys], selected_polys
        else:
          continue
        im = im[ymin:ymax + 1, xmin:xmax + 1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return im, polys, tags, selected_polys
    
    return im, polys, tags, np.array(range(len(polys)))  
    
def shrink_poly(poly, r):
'''
 Copied from anther implmentations
'''
R = 0.3
if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) >  np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
    theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
    poly[0][0] += R * r[0] * np.cos(theta)
    poly[0][1] += R * r[0] * np.sin(theta)
    poly[1][0] -= R * r[1] * np.cos(theta)
    poly[1][1] -= R * r[1] * np.sin(theta)

    theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
    poly[3][0] += R * r[3] * np.cos(theta)
    poly[3][1] += R * r[3] * np.sin(theta)
    poly[2][0] -= R * r[2] * np.cos(theta)
    poly[2][1] -= R * r[2] * np.sin(theta)

    theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
    poly[0][0] += R * r[0] * np.sin(theta)
    poly[0][1] += R * r[0] * np.cos(theta)
    poly[3][0] -= R * r[3] * np.sin(theta)
    poly[3][1] -= R * r[3] * np.cos(theta)

    theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
    poly[1][0] += R * r[1] * np.sin(theta)
    poly[1][1] += R * r[1] * np.cos(theta)
    poly[2][0] -= R * r[2] * np.sin(theta)
    poly[2][1] -= R * r[2] * np.cos(theta)
else:

    theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
    poly[0][0] += R * r[0] * np.sin(theta)
    poly[0][1] += R * r[0] * np.cos(theta)
    poly[3][0] -= R * r[3] * np.sin(theta)
    poly[3][1] -= R * r[3] * np.cos(theta)

    theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
    poly[1][0] += R * r[1] * np.sin(theta)
    poly[1][1] += R * r[1] * np.cos(theta)
    poly[2][0] -= R * r[2] * np.sin(theta)
    poly[2][1] -= R * r[2] * np.cos(theta)

    theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
    poly[0][0] += R * r[0] * np.cos(theta)
    poly[0][1] += R * r[0] * np.sin(theta)
    poly[1][0] -= R * r[1] * np.cos(theta)
    poly[1][1] -= R * r[1] * np.sin(theta)

    theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
    poly[3][0] += R * r[3] * np.cos(theta)
    poly[3][1] += R * r[3] * np.sin(theta)
    poly[2][0] -= R * r[2] * np.cos(theta)
    poly[2][1] -= R * r[2] * np.sin(theta)   
return poly

def point_dist_to_line(p1, p2, p3):
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

  
def fit_line(p1, p2):
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]  
def line_cross_point(line1, line2):
  """
  Copied from anther implmentations
  """
  if line1[0] != 0 and line1[0] == line2[0]:
        return None
    if line1[0] == 0 and line2[0] == 0:
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1-b2)/(k1-k2)
        y = k1*x + b1
    return np.array([x, y], dtype=np.float32)

def line_verticle(line, point):
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1./line[0], -1, point[1] - (-1/line[0] * point[0])]
    return verticle

def rectangle_from_parallelogram(poly):
    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(np.dot(p1-p0, p3-p0)/(np.linalg.norm(p0-p1) * np.linalg.norm(p3-p0)))
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0-p3):
    
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)
            new_p3 = line_cross_point(p2p3, p2p3_verticle)
            
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)

            new_p1 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p0)

            new_p1 = line_cross_point(p1p2, p1p2_verticle)
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)

            new_p3 = line_cross_point(p0p3, p0p3_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0-p1) > np.linalg.norm(p0-p3):
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p1)
            new_p2 = line_cross_point(p2p3, p2p3_verticle)
            
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)

            new_p0 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)

            new_p0 = line_cross_point(p0p3, p0p3_verticle)
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)

            new_p2 = line_cross_point(p1p2, p1p2_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)  
  
def sort_rectangle(poly):
    
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1])/(poly[p_lowest][0] - poly[p_lowest_right][0]))
        
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle/np.pi * 180 > 45:
        
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi/2 - angle)
        else:
            
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle

def generate_rbox(im_size, polys, tags):
    h, w = im_size
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)

    training_mask = np.ones((h, w), dtype=np.uint8)
    rectangles = []
    for poly_idx, poly_tag in enumerate(zip(polys, tags)):
        poly = poly_tag[0]
        tag = poly_tag[1]

        r = [None, None, None, None]
        for i in range(4):
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                       np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
        # score map
        shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
        cv2.fillPoly(score_map, shrinked_poly, 1)
        cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)
        # if the poly is too small, then ignore it during training
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
        if min(poly_h, poly_w) < FLAGS.min_text_size:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
        if tag:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))
        fitted_parallelograms = []
        for i in range(4):
            p0 = poly[i]
            p1 = poly[(i + 1) % 4]
            p2 = poly[(i + 2) % 4]
            p3 = poly[(i + 3) % 4]
            edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p2[0]]
                else:
                    edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
            else:
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p3[0]]
                else:
                    edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
            # move forward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p2 = line_cross_point(forward_edge, edge_opposite)
            if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
                # across p0
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p0[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
            else:
                # across p3
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p3[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
            new_p0 = line_cross_point(forward_opposite, edge)
            new_p3 = line_cross_point(forward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            # or move backward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p3 = line_cross_point(backward_edge, edge_opposite)
            if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
                # across p1
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p1[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
            else:
                # across p2
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p2[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
            new_p1 = line_cross_point(backward_opposite, edge)
            new_p2 = line_cross_point(backward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
        areas = [Polygon(t).area for t in fitted_parallelograms]
        parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
        # sort thie polygon
        parallelogram_coord_sum = np.sum(parallelogram, axis=1)
        min_coord_idx = np.argmin(parallelogram_coord_sum)
        parallelogram = parallelogram[
            [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

        rectange = rectangle_from_parallelogram(parallelogram)
        rectange, rotate_angle = sort_rectangle(rectange)
        rectangles.append(rectange.flatten())
        p0_rect, p1_rect, p2_rect, p3_rect = rectange
        for y, x in xy_in_poly:
            point = np.array([x, y], dtype=np.float32)
            # top
            geo_map[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
            # right
            geo_map[y, x, 1] = point_dist_to_line(p1_rect, p2_rect, point)
            # down
            geo_map[y, x, 2] = point_dist_to_line(p2_rect, p3_rect, point)
            # left
            geo_map[y, x, 3] = point_dist_to_line(p3_rect, p0_rect, point)
            # angle
            geo_map[y, x, 4] = rotate_angle
    return score_map, geo_map, training_mask, rectangles

def get_project_matrix_and_width(text_polyses, text_tags, target_height=8.0):
    project_matrixes = []
    box_widths = []
    
    for i in range(text_polyses.shape[0]):    
        x1, y1, x2, y2, x3, y3, x4, y4 = text_polyses[i] / 4
        rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
        box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]
        if box_w <= box_h:
            box_w, box_h = box_h, box_w
        mapped_x1, mapped_y1 = (0, 0)
        mapped_x4, mapped_y4 = (0, 8)

        width_box = math.ceil(8 * box_w / box_h)
        width_box = int(min(width_box, 128)) 
        mapped_x2, mapped_y2 = (width_box, 0)
	src_pts = np.float32([(x1, y1), (x2, y2), (x4, y4)])
	dst_pts = np.float32([(mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)])
	affine_matrix = cv2.getAffineTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
	affine_matrix = affine_matrix.flatten()
        project_matrixes.append(affine_matrix)
        box_widths.append(width_box)

    project_matrixes = np.array(project_matrixes)
    box_widths = np.array(box_widths)

    return project_matrixes, box_widths

def generator(input_size=512, batch_size=32,
              background_ratio=0, 
              random_scale=np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2]),
              vis=False):
    image_list = np.array(get_images())
    print('{} training images in {}'.format(
        image_list.shape[0], FLAGS.training_data_path))
    index = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(index)
        images = []
        image_fns = []
        score_maps = []
        geo_maps = []
        training_masks = []

        text_polyses = [] 
        text_tagses = []
        boxes_masks = []

        text_labels = []
        count = 0
        for i in index:
            try:
                im_fn = image_list[i]
                im = cv2.imread(im_fn)
                h, w, _ = im.shape
                child_name = im_fn.replace(os.path.basename(im_fn).split('.')[1], 'txt').split('/')[-1]
                txt_fn = "/home/qz/data/ICDAR15/ch4_training_localization_transcription_gt_rec/" + "gt_" + child_name
                if not os.path.exists(txt_fn):
                    print('text file {} does not exists'.format(txt_fn))
                    continue

                text_polys, text_tags, text_label = load_annoataion(txt_fn) 
                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))
                rd_scale = np.random.choice(random_scale)
                im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
                text_polys *= rd_scale                
                if np.random.rand() < background_ratio: 
                    im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
                    if text_polys.shape[0] > 0:
                        continue
                    new_h, new_w, _ = im.shape
                    max_h_w_i = np.max([new_h, new_w, input_size])
                    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy()
                    im = cv2.resize(im_padded, dsize=(input_size, input_size))
                    score_map = np.zeros((input_size, input_size), dtype=np.uint8)
                    geo_map_channels = 5 if FLAGS.geometry == 'RBOX' else 8
                    geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
                    training_mask = np.ones((input_size, input_size), dtype=np.uint8)
                else:
                    im, text_polys, text_tags, selected_poly = crop_area(im, text_polys, text_tags, crop_background=False)                   
                    if text_polys.shape[0] == 0 or len(text_label) == 0:
                        continue
                    h, w, _ = im.shape
                    new_h, new_w, _ = im.shape
                    max_h_w_i = np.max([new_h, new_w, input_size])
                    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy()
                    im = im_padded
                    # resize the image to input size
                    new_h, new_w, _ = im.shape
                    resize_h = input_size
                    resize_w = input_size
                    im = cv2.resize(im, dsize=(resize_w, resize_h))
                    resize_ratio_3_x = resize_w/float(new_w)
                    resize_ratio_3_y = resize_h/float(new_h)
                    text_polys[:, :, 0] *= resize_ratio_3_x
                    text_polys[:, :, 1] *= resize_ratio_3_y
                    new_h, new_w, _ = im.shape

                    score_map, geo_map, training_mask, rectangles = generate_rbox((new_h, new_w), text_polys, text_tags)
                    text_label = [text_label[i] for i in selected_poly]
                    mask = [not (word == [-1]) for word in text_label]
                    text_label = list(compress(text_label, mask))
                    rectangles = list(compress(rectangles, mask))

                    assert len(text_label) == len(rectangles)
                    if len(text_label) == 0:
                        continue

                    boxes_mask = np.array([count] * len(rectangles))

                    count += 1

                images.append(im[:, :, ::-1].astype(np.float32))
                image_fns.append(im_fn)
                score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))

                text_polyses.append(rectangles)
                boxes_masks.append(boxes_mask)
                text_labels.extend(text_label)
                text_tagses.append(text_tags)

                if len(images) == batch_size:
                    text_polyses = np.concatenate(text_polyses)
                    text_tagses = np.concatenate(text_tagses)
                    transform_matrixes, box_widths = get_project_matrix_and_width(text_polyses, text_tagses)
                    # TODO limit the batch size of recognition 
                    text_labels_sparse = sparse_tuple_from(np.array(text_labels))
                    yield images, image_fns, score_maps, geo_maps, training_masks, transform_matrixes, boxes_masks, box_widths, text_labels_sparse,
                    images = []
                    image_fns = []
                    score_maps = []
                    geo_maps = []
                    training_masks = []
                    text_polyses = [] 
                    text_tagses = []
                    boxes_masks = []
                    text_labels = []
                    count = 0
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
        enqueuer.start(max_queue_size=10, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()
