from flask import Flask, request
from flask import jsonify,make_response
import json
import re
import base64
from io import BytesIO
from io import StringIO,TextIOWrapper
from PIL import Image

import time
import argparse
import torch.backends.cudnn as cudnn
#from dataset_gen import *
from model import *
from common import *
from torchvision.utils import save_image
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
from utils import binvox_rw

from stl import mesh
import sys

import trimesh
import scipy
from trimesh.voxel import *

#######################################################################################################
#first we load the model
parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/data/dudong/PartNet.v0/dataset', help='data root path')
parser.add_argument('--thres', type=float, default=0.2, help='threshold for occupancy estimation and mesh extraction')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--test', type=str, default='test', help='test results path')
parser.add_argument('--cat', type=str, default='Chair')
parser.add_argument('--cuda', type=str, default='0')
parser.add_argument('--spaceSize', type=int, default=128, help='voxel space size for assembly')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

cudnn.benchmark = True

vox_res = 64


generator_model_path = './checkpoint/Chair/generator.pt'

generator_network = PartGenerator()

generator_network.load_state_dict(torch.load(generator_model_path,'cpu'))
generator_network.cuda()
generator_network.eval()

# load assemble model

assemble_model_path = './checkpoint/Chair/assembler.pt'

#create assemble network
assemble_network = PartAssembler()
assemble_network.load_state_dict(torch.load(assemble_model_path,map_location='cpu'))
assemble_network.cuda()
assemble_network.eval()


img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

torch.set_grad_enabled(False)

cached_model_list = []
cached_vox_list = []
cached_pose_list = []
pts = make_3d_grid((-0.5,)*3, (0.5,)*3, (vox_res,)*3).contiguous().view(1, -1, 3)
pts = pts.float()
pts = pts.cuda()  

def infer_shape_from_sketch_and_save(img):
    sket_data = img_transform(img).float().contiguous()
    sket_data = sket_data[:3,:,:]
    sket_data = sket_data.unsqueeze(0)
    sket_data = sket_data.cuda()
    pts_occ_val = generator_network.predict(sket_data , pts)
    pts_occ_val = pts_occ_val.contiguous().view(vox_res, vox_res, vox_res).cpu().data.numpy()
    out_vox = pts_occ_val
    #out_vox = np.array(pts_occ_val + (1. - opt.thres), dtype=np.uint8)
    mesh = extract_mesh(pts_occ_val, threshold=opt.thres, n_face_simp=5000)
    
    splitted_mesh = mesh.split()
    if len(splitted_mesh) > 0:
        chosen_id = -1
        max_points = -1
        for i in range(0,len(splitted_mesh)):
            if (splitted_mesh[i].vertices.shape[0] > max_points):
                chosen_id = i
                max_points = splitted_mesh[i].vertices.shape[0]

        mesh = splitted_mesh[chosen_id]
    #trimesh.smoothing.filter_laplacian(mesh)
    #mesh = trimesh.smoothing.filter_laplacian(mesh)
    trimesh.smoothing.filter_taubin(mesh,iterations=10,nu=0.5,lamb=0.9)
    output = mesh.export(file_type='ply',encoding='ascii')    
    return output, out_vox

def infer_shape_from_sketch_and_save_no_mesh(img):
    sket_data = img_transform(img).float().contiguous()
    sket_data = sket_data[:3,:,:]
    sket_data = sket_data.unsqueeze(0)
    sket_data = sket_data.cuda()

    pts_occ_val = generator_network.predict(sket_data , pts)
    pts_occ_val = pts_occ_val.contiguous().view(vox_res, vox_res, vox_res).cpu().data.numpy()
    out_vox = pts_occ_val
    # out_vox = np.array(pts_occ_val + (1. - opt.thres), dtype=np.uint8)
    #mesh = extract_mesh(pts_occ_val, threshold=opt.thres, n_face_simp=5000)


    #output = mesh.export(file_type='ply',encoding='ascii')    
    return out_vox

def infer_pose_from_sketch(full_img, part_img, part_vox):

    full_img_data = img_transform(full_img)[:3,:,:]
    part_img_data = img_transform(part_img)[:3,:,:]

    sket_data = torch.cat((full_img_data, part_img_data),0)
    sket_data = sket_data.unsqueeze(0)

    vox_size = part_vox.shape[0]
    vox_data = np.array(part_vox).reshape((1,1,vox_size,vox_size,vox_size))
    vox_data = torch.from_numpy(vox_data).type(torch.FloatTensor)
    sket_data = sket_data.cuda()
    vox_data = vox_data.cuda()

    pos_pre = assemble_network(sket_data,vox_data)

    pos_pre_np = pos_pre.contiguous().view(-1).cpu().data.numpy() * opt.spaceSize
    
    return pos_pre_np

############################################################################################################


app = Flask('woot-sketch-server')
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers["Access-Control-Allow-Credentials"]="true"
    response.headers["Access-Control-Allow-Methods"]="*"
    response.headers["Access-Control-Allow-Headers"]= "Content-Type,Access-Token"
    response.headers["Access-Control-Expose-Headers"]= "*"
    return response
app.after_request(after_request)

@app.route('/add', methods=['POST'])
def add():
    print(request.json['a'],request.json['b'])
    result = request.json['a'] + request.json['b']
    return str(result)

@app.route('/initModel', methods=['POST'])
def initModel():

    res = make_response(jsonify({}),200)
    return res

image_path_to_save = './images_from_front_end/'

@app.route('/assembleFromImages', methods=['POST'])
def assembleFromImages():
    torch.set_grad_enabled(False)
    request_dict = json.loads(request.data)
    cached_model_list = []
    cached_vox_list = []
    cached_part_pose_list = []
    
    part_data_list = request_dict['part_image']
    whole_image = Image.open(BytesIO(base64.b64decode(request_dict['whole_image'].split(',')[1]))).resize((256,256),Image.ANTIALIAS)
    hx,hy = whole_image.size 
    fin_whole = Image.new('RGBA', whole_image.size, (255,255,255))
    fin_whole.paste(whole_image,(0, 0, hx, hy), whole_image)
    
    # infer
    procesed_img_list = []

    vox_array_list = []
    vox_pose_list = []
    vox_center_list = []
    vox_length_list = []

    for i in range(len(part_data_list)):
        current_url = part_data_list[i].split(',')[1]
        current_url = base64.b64decode(current_url)
        current_url = BytesIO(current_url)
        current_img = Image.open(current_url)
        current_img = current_img.resize((256,256),Image.ANTIALIAS)
        #add a white background
        cx,cy = current_img.size
        p = Image.new('RGBA', current_img.size, (255,255,255))
        p.paste(current_img, (0, 0, cx, cy), current_img)
        procesed_img_list.append(p)
        cur_vox = infer_shape_from_sketch_and_save_no_mesh(p)
        #cached_model_list.append(str(cur_mesh_bit,encoding='ascii'))
        cached_vox_list.append(cur_vox)

        vox_array_list.append(cur_vox.tolist())
    
    #calculate the pose
    for i in range(len(cached_vox_list)):
        current_pose = infer_pose_from_sketch(fin_whole, procesed_img_list[i],cached_vox_list[i])
        cached_part_pose_list.append(current_pose)
        #vox_pose_list.append(current_pose.tolist())
    
    #part_pos_to_list = [t.tolist() for t in cached_part_pose_list]

    #start to assemble 
    whole_vox = np.zeros((opt.spaceSize, opt.spaceSize, opt.spaceSize), dtype=np.uint8)
    #print("part num ",cached_part_pose_list)
    center_arr = []
    part_center_arr = []
    scale_ratio_arr = []
    voxel_to_send = []

    current_mesh_list = []
    for i in range(len(cached_part_pose_list)):
        
        part_vox = cached_vox_list[i]
        part_vox = np.array(part_vox, dtype='uint8')
        #print('part vox shape', part_vox.shape)
        #part_vox = np.array(part_vox)

        part_size = part_vox.shape[0]
        part_pos = np.where(part_vox > 0.1)
        
        #print('part pose before',part_pos)
        part_pos = np.array(part_pos).transpose()
        #print('part pose after',part_pos)
        part_bbox_min = np.min(part_pos, axis=0)
        part_bbox_max = np.max(part_pos, axis=0)

        part_center = (part_bbox_min + part_bbox_max) / 2.
        part_scale = np.linalg.norm(part_bbox_max - part_bbox_min) / 2.

        pos_pre = cached_part_pose_list[i]

        center = np.array((pos_pre[0], pos_pre[1], pos_pre[2]), dtype=np.float)
        scale = np.float(pos_pre[3])

        scale_ratio = scale/part_scale
        
        length = (part_bbox_max - part_bbox_min) * scale_ratio
        bbox_min = np.array(np.clip(center - length / 2., a_min=0, a_max=opt.spaceSize-1), dtype=np.int)
        
        length = np.ceil(length).astype(np.int)

        print('b box min max',bbox_min)

        #128 * 128 * 128
          
        tmp_vox = np.zeros((opt.spaceSize, opt.spaceSize, opt.spaceSize), dtype=np.uint8)
        tmp_vox[bbox_min[0]: bbox_min[0] + length[0], bbox_min[1]: bbox_min[1] + length[1],
            bbox_min[2]: bbox_min[2] + length[2]] = 1
        
        tmp_pos = np.where(tmp_vox > 0.1)
        tmp_pos = np.array(tmp_pos, dtype=np.float).transpose()
        tmp_pos_int = np.array(tmp_pos, dtype=np.int)
        
        center_arr.append(center.tolist()) 
        tmp_pos -= center
        
        tmp_pos = tmp_pos/scale_ratio
        scale_ratio_arr.append(scale_ratio)
        tmp_pos += part_center
        part_center_arr.append(part_center.tolist())

        vox_center_list.append(part_center.tolist())
        vox_length_list.append(scale_ratio)

        tmp_pos_part_int = np.array(tmp_pos, dtype=np.int)
        tmp_pos_part_int = np.clip(tmp_pos_part_int, a_min=0, a_max=part_size-1)
        
        current_vox = np.zeros((opt.spaceSize, opt.spaceSize, opt.spaceSize), dtype=np.uint8)
        whole_vox[tmp_pos_int[:, 0], tmp_pos_int[:, 1], tmp_pos_int[:, 2]] += part_vox[
            tmp_pos_part_int[:, 0], tmp_pos_part_int[:, 1], tmp_pos_part_int[:, 2]]
        
        current_vox[tmp_pos_int[:, 0], tmp_pos_int[:, 1], tmp_pos_int[:, 2]] += part_vox[
            tmp_pos_part_int[:, 0], tmp_pos_part_int[:, 1], tmp_pos_part_int[:, 2]]

        voxel_to_send.append(np.array(np.where(current_vox > 0.1)).tolist())

        current_mesh = extract_mesh(current_vox.astype(np.float), threshold=opt.thres, n_face_simp=6000)
        current_mesh_list.append(current_mesh)
        current_mesh_ascii = current_mesh.export(file_type='ply',encoding='ascii')
        cached_model_list.append(str(current_mesh_ascii,encoding='ascii'))


    mesh = extract_mesh(whole_vox.astype(np.float), threshold=opt.thres, n_face_simp=6000)
    
    mesh_ascii = mesh.export(file_type='ply',encoding='ascii')    

    print('fin cached part pose list',cached_part_pose_list)
   #print(cached_model_list)
    # each part pose
    # each 


    ret_dict = {
        'assembled_model': str(mesh_ascii,encoding='ascii'),
         #'each_part_vox': voxel_to_send,
        'each_part_mesh': cached_model_list,
    }

    res = jsonify(ret_dict)
    #make_response(jsonify(ret_dict),200)
    torch.cuda.empty_cache()
    return res


@app.route('/assembleFromImagesNew', methods=['POST'])
def assembleFromImagesNew():
    torch.set_grad_enabled(False)
    request_dict = json.loads(request.data)
    cached_model_list = []
    cached_vox_list = []
    cached_part_pose_list = []
    
    part_data_list = request_dict['part_image']
    whole_image = Image.open(BytesIO(base64.b64decode(request_dict['whole_image'].split(',')[1]))).resize((256,256),Image.ANTIALIAS)
    hx,hy = whole_image.size 
    fin_whole = Image.new('RGBA', whole_image.size, (255,255,255))
    fin_whole.paste(whole_image,(0, 0, hx, hy), whole_image)

    part_vox_list = request_dict['part_vox']
    
    # infer
    procesed_img_list = []

    vox_array_list = []
    vox_pose_list = []
    vox_center_list = []
    vox_length_list = []

    for i in range(len(part_data_list)):
        current_url = part_data_list[i].split(',')[1]
        current_url = base64.b64decode(current_url)
        current_url = BytesIO(current_url)
        current_img = Image.open(current_url)
        current_img = current_img.resize((256,256),Image.ANTIALIAS)
        #add a white background
        cx,cy = current_img.size
        p = Image.new('RGBA', current_img.size, (255,255,255))
        p.paste(current_img, (0, 0, cx, cy), current_img)
        procesed_img_list.append(p)
        xs = np.array(part_vox_list[i][0],dtype=np.int)
        ys = np.array(part_vox_list[i][1],dtype=np.int)
        zs = np.array(part_vox_list[i][2],dtype=np.int)
        v_s = np.array(part_vox_list[i][3],dtype=np.float).reshape(-1)

        #voxres
        cur_vox = np.zeros(shape=(vox_res,vox_res,vox_res),dtype=np.float)
        cur_vox[xs,ys,zs] = v_s
        
        #cur_vox = infer_shape_from_sketch_and_save_no_mesh(p)
        #cached_model_list.append(str(cur_mesh_bit,encoding='ascii'))
        cached_vox_list.append(cur_vox)

        #vox_array_listvox_array_list.append(cur_vox.tolist())
    
    #calculate the pose
    for i in range(len(cached_vox_list)):
        current_pose = infer_pose_from_sketch(fin_whole, procesed_img_list[i],cached_vox_list[i])
        cached_part_pose_list.append(current_pose)
        #vox_pose_list.append(current_pose.tolist())
    
    #part_pos_to_list = [t.tolist() for t in cached_part_pose_list]

    #start to assemble 
    whole_vox = np.zeros((opt.spaceSize, opt.spaceSize, opt.spaceSize), dtype=np.uint8)
    #print("part num ",cached_part_pose_list)
    center_arr = []
    part_center_arr = []
    scale_ratio_arr = []
    voxel_to_send = []

    cleaned_smoothed_mesh = []
    cleaned_smoothed_face = []
    cleaned_smoothed_points = []
    base_vertices_num = 0

    for i in range(len(cached_part_pose_list)):
        
        part_vox = cached_vox_list[i]
        part_vox = np.array(part_vox, dtype='float')
        #print('part vox shape', part_vox.shape)
        #part_vox = np.array(part_vox)

        part_size = part_vox.shape[0]
        part_pos = np.where(part_vox > 0.01)
        
        #print('part pose before',part_pos)
        part_pos = np.array(part_pos).transpose()
        #print('part pose after',part_pos)
        part_bbox_min = np.min(part_pos, axis=0)
        part_bbox_max = np.max(part_pos, axis=0)

        part_center = (part_bbox_min + part_bbox_max) / 2.
        part_scale = np.linalg.norm(part_bbox_max - part_bbox_min) / 2.

        pos_pre = cached_part_pose_list[i]

        center = np.array((pos_pre[0], pos_pre[1], pos_pre[2]), dtype=np.float)
        scale = np.float(pos_pre[3])

        scale_ratio = scale/part_scale
        
        length = (part_bbox_max - part_bbox_min) * scale_ratio
        bbox_min = np.array(np.clip(center - length / 2., a_min=0, a_max=opt.spaceSize-1), dtype=np.int)
        
        length = np.ceil(length).astype(np.int)

        #print('b box min max',bbox_min)

        #128 * 128 * 128
          
        tmp_vox = np.zeros((opt.spaceSize, opt.spaceSize, opt.spaceSize), dtype=np.uint8)
        tmp_vox[bbox_min[0]: bbox_min[0] + length[0], bbox_min[1]: bbox_min[1] + length[1],
            bbox_min[2]: bbox_min[2] + length[2]] = 1
        
        tmp_pos = np.where(tmp_vox > 0.01)
        tmp_pos = np.array(tmp_pos, dtype=np.float).transpose()
        tmp_pos_int = np.array(tmp_pos, dtype=np.int)
        
        center_arr.append(center.tolist()) 
        tmp_pos -= center
        
        tmp_pos = tmp_pos/scale_ratio
        scale_ratio_arr.append(scale_ratio)
        tmp_pos += part_center
        part_center_arr.append(part_center.tolist())

        vox_center_list.append(part_center.tolist())
        vox_length_list.append(scale_ratio)

        tmp_pos_part_int = np.array(tmp_pos, dtype=np.int)
        tmp_pos_part_int = np.clip(tmp_pos_part_int, a_min=0, a_max=part_size-1)
        
        current_vox = np.zeros((opt.spaceSize, opt.spaceSize, opt.spaceSize), dtype=np.float)
        #whole_vox[tmp_pos_int[:, 0], tmp_pos_int[:, 1], tmp_pos_int[:, 2]] += part_vox[
        #    tmp_pos_part_int[:, 0], tmp_pos_part_int[:, 1], tmp_pos_part_int[:, 2]]
        
        current_vox[tmp_pos_int[:, 0], tmp_pos_int[:, 1], tmp_pos_int[:, 2]] += part_vox[
            tmp_pos_part_int[:, 0], tmp_pos_part_int[:, 1], tmp_pos_part_int[:, 2]]

        #voxel_to_send.append(np.array(np.where(current_vox > 0.1)).tolist())

        current_mesh = extract_mesh(current_vox.astype(np.float), threshold=opt.thres, n_face_simp=5000)
        
        splitted_mesh = current_mesh.split()
        chosen_id = -1
        max_points = -1
        
        if(len(splitted_mesh)>0):
            for i in range(0,len(splitted_mesh)):
                if (splitted_mesh[i].vertices.shape[0] > max_points):
                    chosen_id = i
                    max_points = splitted_mesh[i].vertices.shape[0]

            current_mesh = splitted_mesh[chosen_id]
        
        trimesh.smoothing.filter_taubin(current_mesh,iterations=10,nu=0.5,lamb=0.9)
        
        current_mesh_ascii = current_mesh.export(file_type='ply',encoding='ascii')
        cached_model_list.append(str(current_mesh_ascii,encoding='ascii'))
        cleaned_smoothed_mesh.append(current_mesh)
        cleaned_smoothed_face += (current_mesh.faces + base_vertices_num).tolist()
        cleaned_smoothed_points += current_mesh.vertices.tolist()
        base_vertices_num += current_mesh.vertices.shape[0]

    #interfaces.blender.boolean(cleaned_smoothed_mesh,operation='union', debug=False)

    union_mesh = trimesh.Trimesh(vertices=np.array(cleaned_smoothed_points),faces=np.array(cleaned_smoothed_face))
    """
    union_mesh.export('meshunion.ply')
    
    fin_whole_vox = -1

    cur_pitch = 1.0/128
    
    occupancy_points = []
    b_min = []
    b_max = []
    trimesh_mesh = []

    for i in range(len(cleaned_smoothed_mesh)):
        new_mesh = cleaned_smoothed_mesh[i]
        new_mesh.remove_degenerate_faces()

        trimesh.repair.fill_holes(new_mesh)

        c_max = np.max(new_mesh.vertices,0)
        c_min = np.min(new_mesh.vertices,0)
        b_min.append(c_min.tolist())
        b_max.append(c_max.tolist())
    
        new_vox = new_mesh.voxelized(pitch=cur_pitch)
        occupancy_points = occupancy_points + new_vox.indices_to_points(new_vox.sparse_indices).tolist()
        trimesh_mesh.append(new_mesh)
    
    b_min = np.min(np.array(b_min),0)
    b_max = np.max(np.array(b_max),0)
    
    b_mid = (b_min + b_max )*0.5

    occupancy_points = np.array(occupancy_points)

    #print("occupancy points shape",np.max((occupancy_points),0),np.min((occupancy_points),0))

    occupancy_points += 0.5 
    occupancy_points *= opt.spaceSize
    occupancy_points_int = np.array(occupancy_points, dtype=np.int)
    occupancy_points_int = np.clip(occupancy_points_int, a_min=0, a_max=opt.spaceSize-1)
   
   
    whole_occ_grid =  np.zeros((opt.spaceSize, opt.spaceSize, opt.spaceSize), dtype=np.float)
    
    whole_occ_grid[occupancy_points_int[:,0],occupancy_points_int[:,1],occupancy_points_int[:,2]] += 0.5

    fin_mesh = extract_mesh(whole_occ_grid.astype(np.float), threshold=opt.thres)
    #trimesh.smoothing.filter_taubin(fin_mesh)


    n_min, n_max = np.min(fin_mesh.vertices,0), np.max(fin_mesh.vertices,0)
    
    fin_mesh.vertices *= (b_max-b_min) / (n_max-n_min)
    
    n_min, n_max = np.min(fin_mesh.vertices,0), np.max(fin_mesh.vertices,0)

    n_mid = (n_min + n_max) * 0.5

    fin_mesh.vertices += b_mid - n_mid

    #fin_mesh.export('full_mesh.ply') 
    trimesh.smoothing.filter_humphrey(fin_mesh)
    """

    fin_mesh_ascii = union_mesh.export(file_type='ply',encoding='ascii')    


    ret_dict = {
        'assembled_model': str(fin_mesh_ascii,encoding='ascii'),
        'each_part_mesh': cached_model_list,
    }

    res = jsonify(ret_dict)
    #make_response(jsonify(ret_dict),200)
    torch.cuda.empty_cache()
    return res


"""
@app.route('/assembleFromImagesNew', methods=['POST'])
def assembleFromImagesNew():
    torch.set_grad_enabled(False)
    request_dict = json.loads(request.data)
    cached_model_list = []
    cached_vox_list = []
    cached_part_pose_list = []
    
    part_data_list = request_dict['part_image']
    whole_image = Image.open(BytesIO(base64.b64decode(request_dict['whole_image'].split(',')[1]))).resize((256,256),Image.ANTIALIAS)
    hx,hy = whole_image.size 
    fin_whole = Image.new('RGBA', whole_image.size, (255,255,255))
    fin_whole.paste(whole_image,(0, 0, hx, hy), whole_image)

    part_vox_list = request_dict['part_vox']
    
    # infer
    procesed_img_list = []

    vox_array_list = []
    vox_pose_list = []
    vox_center_list = []
    vox_length_list = []

    for i in range(len(part_data_list)):
        current_url = part_data_list[i].split(',')[1]
        current_url = base64.b64decode(current_url)
        current_url = BytesIO(current_url)
        current_img = Image.open(current_url)
        current_img = current_img.resize((256,256),Image.ANTIALIAS)
        #add a white background
        cx,cy = current_img.size
        p = Image.new('RGBA', current_img.size, (255,255,255))
        p.paste(current_img, (0, 0, cx, cy), current_img)
        procesed_img_list.append(p)
        xs = np.array(part_vox_list[i][0],dtype=np.int)
        ys = np.array(part_vox_list[i][1],dtype=np.int)
        zs = np.array(part_vox_list[i][2],dtype=np.int)
        v_s = np.array(part_vox_list[i][3],dtype=np.float).reshape(-1)

        #voxres
        cur_vox = np.zeros(shape=(vox_res,vox_res,vox_res),dtype=np.float)
        cur_vox[xs,ys,zs] = v_s
        
        cached_vox_list.append(cur_vox)

        #vox_array_listvox_array_list.append(cur_vox.tolist())
    
    #calculate the pose
    for i in range(len(cached_vox_list)):
        current_pose = infer_pose_from_sketch(fin_whole, procesed_img_list[i],cached_vox_list[i])
        cached_part_pose_list.append(current_pose)
        #vox_pose_list.append(current_pose.tolist())
    
    #part_pos_to_list = [t.tolist() for t in cached_part_pose_list]

    #start to assemble 
    whole_vox = np.zeros((opt.spaceSize, opt.spaceSize, opt.spaceSize), dtype=np.float)
    #print("part num ",cached_part_pose_list)
    center_arr = []
    part_center_arr = []
    scale_ratio_arr = []
    voxel_to_send = []

    cleaned_smoothed_mesh = []
    org_vox = []

    with_noise_max = []
    with_noise_min = []


    for i in range(len(cached_part_pose_list)):
        
        part_vox = cached_vox_list[i]
        part_vox = np.array(part_vox, dtype='float')

        part_size = part_vox.shape[0]
        part_pos = np.where(part_vox > 0.01)
        
        #print('part pose before',part_pos)
        part_pos = np.array(part_pos).transpose()
        #print('part pose after',part_pos)
        part_bbox_min = np.min(part_pos, axis=0)
        part_bbox_max = np.max(part_pos, axis=0)

        part_center = (part_bbox_min + part_bbox_max) / 2.
        part_scale = np.linalg.norm(part_bbox_max - part_bbox_min) / 2.

        pos_pre = cached_part_pose_list[i]

        center = np.array((pos_pre[0], pos_pre[1], pos_pre[2]), dtype=np.float)
        scale = np.float(pos_pre[3])

        scale_ratio = scale/part_scale
        
        length = (part_bbox_max - part_bbox_min) * scale_ratio
        bbox_min = np.array(np.clip(center - length / 2., a_min=0, a_max=opt.spaceSize-1), dtype=np.int)
        
        length = np.ceil(length).astype(np.int)

          
        tmp_vox = np.zeros((opt.spaceSize, opt.spaceSize, opt.spaceSize), dtype=np.uint8)
        tmp_vox[bbox_min[0]: bbox_min[0] + length[0], bbox_min[1]: bbox_min[1] + length[1],
            bbox_min[2]: bbox_min[2] + length[2]] = 1
        
        tmp_pos = np.where(tmp_vox > 0.01)
        tmp_pos = np.array(tmp_pos, dtype=np.float).transpose()
        tmp_pos_int = np.array(tmp_pos, dtype=np.int)
        
        center_arr.append(center.tolist()) 
        tmp_pos -= center
        
        tmp_pos = tmp_pos/scale_ratio
        scale_ratio_arr.append(scale_ratio)
        tmp_pos += part_center
        part_center_arr.append(part_center.tolist())

        vox_center_list.append(part_center.tolist())
        vox_length_list.append(scale_ratio)

        tmp_pos_part_int = np.array(tmp_pos, dtype=np.int)
        tmp_pos_part_int = np.clip(tmp_pos_part_int, a_min=0, a_max=part_size-1)
        
        current_vox = np.zeros((opt.spaceSize, opt.spaceSize, opt.spaceSize), dtype=np.float)
        
        current_vox[tmp_pos_int[:, 0], tmp_pos_int[:, 1], tmp_pos_int[:, 2]] += part_vox[
            tmp_pos_part_int[:, 0], tmp_pos_part_int[:, 1], tmp_pos_part_int[:, 2]]

        whole_vox[tmp_pos_int[:, 0], tmp_pos_int[:, 1], tmp_pos_int[:, 2]] += part_vox[
            tmp_pos_part_int[:, 0], tmp_pos_part_int[:, 1], tmp_pos_part_int[:, 2]]
        
        org_vox.append(current_vox)
        #voxel_to_send.append(np.array(np.where(current_vox > 0.1)).tolist())

        current_mesh = extract_mesh(current_vox.astype(np.float), threshold=opt.thres, n_face_simp=5000)
        
        cur_with_noise_max = np.max(current_mesh.vertices,0)
        cur_with_noise_min = np.min(current_mesh.vertices,0)
        with_noise_max.append(cur_with_noise_max)
        with_noise_min.append(cur_with_noise_min)

        splitted_mesh = current_mesh.split()
        chosen_id = -1
        max_points = -1
        
        if(len(splitted_mesh)>0):
            for i in range(0,len(splitted_mesh)):
                if (splitted_mesh[i].vertices.shape[0] > max_points):
                    chosen_id = i
                    max_points = splitted_mesh[i].vertices.shape[0]

            current_mesh = splitted_mesh[chosen_id]
        
        trimesh.smoothing.filter_taubin(current_mesh,iterations=20,nu=0.5,lamb=0.9)
        
        current_mesh_ascii = current_mesh.export(file_type='ply',encoding='ascii')
        cached_model_list.append(str(current_mesh_ascii,encoding='ascii'))
        cleaned_smoothed_mesh.append(current_mesh)


    
    #merged_mesh = trimesh.creation(current_mesh_list)
    #merged_mesh.export("yahoo.ply")

    whole_mesh = extract_mesh(whole_vox.astype(np.float), threshold=opt.thres, n_face_simp=10000)

    splitted_mesh =  whole_mesh.split()
    chosen_id = -1
    max_points = -1


    if(len(splitted_mesh)>0):
        for i in range(0,len(splitted_mesh)):
            if (splitted_mesh[i].vertices.shape[0] > max_points):
                chosen_id = i
                max_points = splitted_mesh[i].vertices.shape[0]

        whole_mesh = splitted_mesh[chosen_id]

    fin_mesh_ascii = whole_mesh.export(file_type='ply',encoding='ascii')

    ret_dict = {
        'assembled_model': str(fin_mesh_ascii,encoding='ascii'),
        'each_part_mesh': cached_model_list,
    }

    res = jsonify(ret_dict)
    #make_response(jsonify(ret_dict),200)
    torch.cuda.empty_cache()
    return res
"""
@app.route('/inferAllParts',methods=['POST'])
def inferAllParts():
    torch.set_grad_enabled(False)
    request_dict = json.loads(request.data)
    cached_model_list = []
    cached_vox_list = []
    part_data_list = request_dict['part_image']
    
    current_vox_array_list = []
    
    for i in range(len(part_data_list)):
        current_url = part_data_list[i].split(',')[1]
        current_url = base64.b64decode(current_url)
        current_url = BytesIO(current_url)
        current_img = Image.open(current_url)
        #print('channel num',current_img.)
        current_img = current_img.resize((256,256),Image.ANTIALIAS)
        #add a white background
        cx,cy = current_img.size
        p = Image.new('RGBA', current_img.size, (255,255,255))

        p.paste(current_img, (0, 0, cx, cy), current_img)

        cur_mesh_bit, cur_vox = infer_shape_from_sketch_and_save(p)

        cached_model_list.append(str(cur_mesh_bit,encoding='ascii'))
        cur_idx = np.where(cur_vox>0.01)
        cur_vox_value = cur_vox[cur_idx[0],cur_idx[1],cur_idx[2]]
        
        
        cached_vox_list.append([cur_idx[0].tolist(), cur_idx[1].tolist(),cur_idx[2].tolist(),cur_vox_value.tolist()])
        #print(np.where(cur_vox>0))

        
   #print(cached_model_list)
    
    ret_dict = {
        'all_parts': cached_model_list,
        'all_voxes':cached_vox_list,
    }

    res = jsonify(ret_dict)
    #make_response(jsonify(ret_dict),200)
    torch.cuda.empty_cache()
    return res


@app.route('/changeModelType',methods=['POST'])
def changeModelType():
    request_dict = json.loads(request.data)
    next_model_type = request_dict['modelType']
    if(True):
       
        generator_model_path = './checkpoint/'+ next_model_type + '/generator.pt'

        generator_network.load_state_dict(torch.load(generator_model_path,'cpu'))
        generator_network.cuda()
        generator_network.eval()

        # load assemble model

        assemble_model_path = './checkpoint/'+ next_model_type + '/assembler.pt'

        assemble_network.load_state_dict(torch.load(assemble_model_path,map_location='cpu'))
        assemble_network.cuda()
        assemble_network.eval()


    ret_dict = {
        'spaceholder':'heihei'
    }
    res = jsonify(ret_dict)
    return res

@app.route('/generateTransformedResults',methods=['POST'])
def generateTransformedResults():
    request_dict = json.loads(request.data)
    print(request_dict.keys())
    mesh_arr = []
    tranform_arr = []
    scale_arr = []
    trimesh_mesh = []
    part_vox_info_arr = []
    #print(request_dict['scale_arr'],request_dict['transform_arr'])
    
    for i in range(len(request_dict['scale_arr'])):
        scale_arr.append(np.array([request_dict['scale_arr'][i][0],request_dict['scale_arr'][i][1],request_dict['scale_arr'][i][2]]) )
        tranform_arr.append(np.array([request_dict['transform_arr'][i][0],request_dict['transform_arr'][i][1],request_dict['transform_arr'][i][2]]))
        mesh_arr.append(request_dict['mesh_string_arr'][i])
        
    
    fin_whole_vox = -1

    cur_pitch = 1.0/128
    
    occupancy_points = []
    b_min = []
    b_max = []

    cleaned_smoothed_face = []
    cleaned_smoothed_points = []
    base_vertices_num = 0

    for i in range(len(mesh_arr)):
        
        new_mesh = trimesh.load(file_obj= BytesIO(mesh_arr[i].encode(encoding='utf-8')),file_type='ply')
        new_mesh.remove_degenerate_faces()
        #print('mesh shape',i,new_mesh.vertices.shape,new_mesh.is_watertight)
        trimesh.repair.fill_holes(new_mesh)
        #print(scale_arr[i],tranform_arr[i])
        new_mesh.vertices[:,0] *= scale_arr[i][0]
        new_mesh.vertices[:,1] *= scale_arr[i][1]
        new_mesh.vertices[:,2] *= scale_arr[i][2]
        
        new_mesh.vertices[:,0] += tranform_arr[i][0]
        new_mesh.vertices[:,1] += tranform_arr[i][1]
        new_mesh.vertices[:,2] += tranform_arr[i][2]

        c_max = np.max(new_mesh.vertices,0)
        c_min = np.min(new_mesh.vertices,0)
        b_min.append(c_min.tolist())
        b_max.append(c_max.tolist())
    
        new_vox = new_mesh.voxelized(pitch=cur_pitch)
        print(new_vox.scale)
        occupancy_points = occupancy_points + new_vox.indices_to_points(new_vox.sparse_indices).tolist()

        trimesh_mesh.append(new_mesh)
        
        #cleaned_smoothed_mesh.append(current_mesh)
        cleaned_smoothed_face += (new_mesh.faces + base_vertices_num).tolist()
        cleaned_smoothed_points += new_mesh.vertices.tolist()
        base_vertices_num += new_mesh.vertices.shape[0]
    
    union_mesh = trimesh.Trimesh(vertices=np.array(cleaned_smoothed_points),faces=np.array(cleaned_smoothed_face))
    #union_vox = trimesh.voxel.creation.voxelize(union_mesh,pitch=cur_pitch)
    fin_mesh_ascii = union_mesh.export(file_type='ply',encoding='ascii')    
    #union_vox.marching_cubes.export('fin_marching_cubes.ply')
    ret_dict = {
        'assembled_model': str(fin_mesh_ascii,encoding='ascii'),
        'each_part_mesh': [str(t.export(file_type='ply',encoding='ascii') ,encoding='ascii') for t in trimesh_mesh]
    }
    res = jsonify(ret_dict)

    return res


"""
@app.route('/generateTransformedResults',methods=['POST'])
def generateTransformedResults():
    request_dict = json.loads(request.data)
    print(request_dict.keys())
    mesh_arr = []
    tranform_arr = []
    scale_arr = []
    trimesh_mesh = []
    part_vox_info_arr = []
    #print(request_dict['scale_arr'],request_dict['transform_arr'])
    
    for i in range(len(request_dict['scale_arr'])):
        scale_arr.append(np.array([request_dict['scale_arr'][i][0],request_dict['scale_arr'][i][1],request_dict['scale_arr'][i][2]]) )
        tranform_arr.append(np.array([request_dict['transform_arr'][i][0],request_dict['transform_arr'][i][1],request_dict['transform_arr'][i][2]]))
        mesh_arr.append(request_dict['mesh_string_arr'][i])
        
    
    fin_whole_vox = -1

    cur_pitch = 1.0/128
    
    occupancy_points = []
    b_min = []
    b_max = []

    for i in range(len(mesh_arr)):
        
        new_mesh = trimesh.load(file_obj= BytesIO(mesh_arr[i].encode(encoding='utf-8')),file_type='ply')
        new_mesh.remove_degenerate_faces()
        #print('mesh shape',i,new_mesh.vertices.shape,new_mesh.is_watertight)
        trimesh.repair.fill_holes(new_mesh)
        #print(scale_arr[i],tranform_arr[i])
        new_mesh.vertices[:,0] *= scale_arr[i][0]
        new_mesh.vertices[:,1] *= scale_arr[i][1]
        new_mesh.vertices[:,2] *= scale_arr[i][2]
        
        new_mesh.vertices[:,0] += tranform_arr[i][0]
        new_mesh.vertices[:,1] += tranform_arr[i][1]
        new_mesh.vertices[:,2] += tranform_arr[i][2]

        c_max = np.max(new_mesh.vertices,0)
        c_min = np.min(new_mesh.vertices,0)
        b_min.append(c_min.tolist())
        b_max.append(c_max.tolist())
    
        new_vox = new_mesh.voxelized(pitch=cur_pitch)
        print(new_vox.scale)
        occupancy_points = occupancy_points + new_vox.indices_to_points(new_vox.sparse_indices).tolist()
        #new_mesh.export(str(i)+'.ply')      
        #print('mesh shape',i,new_mesh.vertices.shape,new_mesh.is_watertight)
        trimesh_mesh.append(new_mesh)
    
    #fin_whole_vox = trimesh.voxel.VoxelGrid(encoding=trimesh.voxel.ops.sparse_to_matrix(np.array(occupancy_points)))
    
    b_min = np.min(np.array(b_min),0)
    b_max = np.max(np.array(b_max),0)
    
    b_mid = (b_min + b_max )*0.5

    occupancy_points = np.array(occupancy_points)

    #print("occupancy points shape",np.max((occupancy_points),0),np.min((occupancy_points),0))

    occupancy_points += 0.5 
    occupancy_points *= opt.spaceSize
    occupancy_points_int = np.array(occupancy_points, dtype=np.int)
    occupancy_points_int = np.clip(occupancy_points_int, a_min=0, a_max=opt.spaceSize-1)


    whole_occ_grid =  np.zeros((opt.spaceSize, opt.spaceSize, opt.spaceSize), dtype=np.uint8)
    
    whole_occ_grid[occupancy_points_int[:,0],occupancy_points_int[:,1],occupancy_points_int[:,2]] += 1

    fin_mesh = extract_mesh(whole_occ_grid.astype(np.float), threshold=opt.thres)
    
    trimesh.smoothing.filter_taubin(fin_mesh,iterations=5)

    n_min, n_max = np.min(fin_mesh.vertices,0), np.max(fin_mesh.vertices,0)
    
    fin_mesh.vertices *= (b_max-b_min) / (n_max-n_min)
    
    n_min, n_max = np.min(fin_mesh.vertices,0), np.max(fin_mesh.vertices,0)

    n_mid = (n_min + n_max) * 0.5

    fin_mesh.vertices += b_mid - n_mid

    #fin_mesh.export('full_mesh.ply') 
    

    fin_mesh_ascii = fin_mesh.export(file_type='ply',encoding='ascii')    

    
    #print(cached_model_list)
    # each part pose
    # each 
    ret_dict = {
        'assembled_model': str(fin_mesh_ascii,encoding='ascii'),
        'each_part_mesh': [str(t.export(file_type='ply',encoding='ascii') ,encoding='ascii') for t in trimesh_mesh]
    }
    res = jsonify(ret_dict)

    return res
"""


if __name__ == '__main__':
    app.run(host='localhost', port=11451, debug=True)