# /******************************************
# *MIT License
# *
# *Copyright (c) [2023] [Giuseppe Sorrentino, Marco Venere, Eleonora D'Arnese, Davide Conficconi, Isabella Poles, Marco Domenico Santambrogio]
# *
# *Permission is hereby granted, free of charge, to any person obtaining a copy
# *of this software and associated documentation files (the "Software"), to deal
# *in the Software without restriction, including without limitation the rights
# *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# *copies of the Software, and to permit persons to whom the Software is
# *furnished to do so, subject to the following conditions:
# *
# *The above copyright notice and this permission notice shall be included in all
# *copies or substantial portions of the Software.
# *
# *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# *SOFTWARE.
# ******************************************/
from email.mime import image

from pickletools import uint8
from re import X
from tokenize import Double
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import torch
import kornia
from PIL import Image
import torchvision.transforms as transforms
import kornia.geometry.transform as IR

"""
this is the function that estimates the initial parameters for the computation.
The idea is averaging the 2D moments."""

def estimate_initial(Ref_uint8s,Flt_uint8s, params, volume):
    tot_flt_avg_10 = 0
    tot_flt_avg_01 = 0
    tot_flt_mu_20 = 0
    tot_flt_mu_02 = 0
    tot_flt_mu_11 = 0
    tot_ref_avg_10 = 0
    tot_ref_avg_01 = 0
    tot_ref_mu_20 = 0
    tot_ref_mu_02 = 0
    tot_ref_mu_11 = 0
    tot_params1 = 0
    tot_params2 = 0
    tot_roundness = 0
    for i in range(0, volume):
        Ref_uint8 = Ref_uint8s[i, :, :]
        Flt_uint8 = Flt_uint8s[i, :, :]
        X = Ref_uint8.numpy()
        Y = Flt_uint8.numpy()
        ref_mom = cv2.moments(X)
        flt_mom = cv2.moments(Y)
        flt_avg_10 = flt_mom['m10']/flt_mom['m00']
        flt_avg_01 = flt_mom['m01']/flt_mom['m00']
        flt_mu_20 = (flt_mom['m20']/flt_mom['m00']*1.0)-(flt_avg_10*flt_avg_10)
        flt_mu_02 = (flt_mom['m02']/flt_mom['m00']*1.0)-(flt_avg_01*flt_avg_01)
        flt_mu_11 = (flt_mom['m11']/flt_mom['m00']*1.0)-(flt_avg_01*flt_avg_10)
        ref_avg_10 = ref_mom['m10']/ref_mom['m00']
        ref_avg_01 = ref_mom['m01']/ref_mom['m00']
        ref_mu_20 = (ref_mom['m20']/ref_mom['m00']*1.0)-(ref_avg_10*ref_avg_10)
        ref_mu_02 = (ref_mom['m02']/ref_mom['m00']*1.0)-(ref_avg_01*ref_avg_01)
        ref_mu_11 = (ref_mom['m11']/ref_mom['m00']*1.0)-(ref_avg_01*ref_avg_10)
        params1 = ref_mom['m10']/ref_mom['m00']-flt_mom['m10']/flt_mom['m00']
        params2 = ref_mom['m01']/ref_mom['m00'] - flt_mom['m01']/flt_mom['m00']
        roundness=(flt_mom['m20']/flt_mom['m00']) / (flt_mom['m02']/flt_mom['m00'])
        tot_flt_avg_10 += flt_avg_10
        tot_flt_avg_01 += flt_avg_01
        tot_flt_mu_20 += flt_mu_20
        tot_flt_mu_02 += flt_mu_02
        tot_flt_mu_11 += flt_mu_11
        tot_ref_avg_10 += ref_avg_10
        tot_ref_avg_01 += ref_avg_01
        tot_ref_mu_20 += ref_mu_20
        tot_ref_mu_02 += ref_mu_02
        tot_ref_mu_11 += ref_mu_11
        tot_params1 += params1
        tot_params2 += params2
        tot_roundness += roundness
    tot_flt_avg_10 = tot_flt_avg_10/volume
    tot_flt_avg_01 = tot_flt_avg_01/volume
    tot_flt_mu_20 = tot_flt_mu_20/volume
    tot_flt_mu_02 = tot_flt_mu_02/volume
    tot_flt_mu_11 = tot_flt_mu_11/volume
    tot_ref_avg_10 = tot_ref_avg_10/volume
    tot_ref_avg_01 = tot_ref_avg_01/volume
    tot_ref_mu_20 = tot_ref_mu_20/volume
    tot_ref_mu_02 = tot_ref_mu_02/volume
    tot_ref_mu_11 = tot_ref_mu_11/volume
    tot_params1 = tot_params1/volume
    tot_params2 = tot_params2/volume
    tot_roundness = tot_roundness/volume

    params[0][3] = tot_params1
    params[1][3] = tot_params2
    rho_flt=0.5*math.atan((2.0*tot_flt_mu_11)/(tot_flt_mu_20-tot_flt_mu_02))
    rho_ref=0.5*math.atan((2.0*tot_ref_mu_11)/(tot_ref_mu_20-tot_ref_mu_02))
    delta_rho=rho_ref-rho_flt
#since the matrix we want to create is an affine matrix, the initial parameters have been prepared as a "particular" affine, the similarity matrix.
    if math.fabs(tot_roundness-1.0)>=0.3:
        params[0][0]= math.cos(delta_rho)
        params[0][1] = -math.sin(delta_rho)
        params[1][0] = math.sin(delta_rho)
        params[1][1] = math.cos(delta_rho)
    else:
        params[0][0]= 1.0
        params[0][1] = 0.0
        params[1][0] = 0.0
        params[1][1] = 1.0
    params[2][2] = 1
    params[0][2] = params[0][3] = 0
    params[2][0] = params[2][1] = 0
    
    return params

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

"""
this function aims to perform transformations. In particular, it prepares the data for the warp_affine3D performed by the kornia libraries.
"""
def transform(images, par, volume):
    tensor3D = torch.reshape(images, (1,1,volume,512, 512)).type(torch.DoubleTensor)
    
    par_tensor = torch.tensor(np.array([par]),dtype=torch.float64)
    par_tensor = torch.reshape(par_tensor,(1,3,4))
    newTensor3D = IR.warp_affine3d(tensor3D, par_tensor, dsize=(volume, 512, 512),align_corners=True)

    newTensor3D = torch.reshape(newTensor3D, (volume,1,512, 512))
    newArray3D = kornia.tensor_to_image(newTensor3D.byte())
    newArray3D = np.reshape(newArray3D,(volume,512,512))

    return(newArray3D)


    
    


def turnaindre(vec):
    mat=np.zeros((2,3))
    mat[0][0]=vec[2]
    mat[0][1]=vec[3]
    mat[0][2]=vec[0]
    mat[1][0]=vec[4]
    mat[1][1]=vec[5]
    mat[1][2]=vec[1]
    return (mat)


def to_matrix_complete(vector_params):
    """
        vector_params contains tx, ty, tz for translation on x, y and z axes
        and cosine of phi, theta, psi for rotations around x, y, and z axes.
    """
    mat_params=np.zeros((3,4))
    mat_params[0][3]=vector_params[0] 
    mat_params[1][3]=vector_params[1] 
    mat_params[2][3]=vector_params[2]
    cos_phi = vector_params[3]
    cos_theta = vector_params[4]
    cos_psi = vector_params[5]
    if cos_phi > 1 or cos_phi < -1:
        cos_phi = 1
    if cos_theta > 1 or cos_theta < -1:
        cos_theta = 1
    if cos_psi > 1 or cos_psi < -1:
        cos_psi = 1
    sin_phi = -np.sqrt(1-(cos_phi**2))
    sin_theta = -np.sqrt(1-(cos_theta**2))
    sin_psi = -np.sqrt(1-(cos_psi**2))
    sin_theta_sin_psi = sin_theta * sin_psi
    sin_theta_cos_psi = sin_theta * cos_psi
    cos_theta_cos_psi = cos_theta * cos_psi
    cos_theta_sin_psi = cos_theta * sin_psi
    cos_phi_sin_phi = cos_phi * sin_phi
    cos_phi_cos_psi = cos_phi * cos_psi
    sin_phi_cos_theta = sin_phi * cos_theta
    sin_phi_sin_psi = sin_phi * sin_psi
    cos_phi_cos_theta = cos_phi * cos_theta
    sin_phi_cos_psi = sin_phi * cos_psi
    mat_params[0][0] = cos_theta_cos_psi
    mat_params[1][0] = cos_theta_sin_psi
    mat_params[2][0] = -sin_theta
    mat_params[0][1] = -cos_phi_sin_phi + sin_phi * sin_theta_cos_psi
    mat_params[1][1] = cos_phi_cos_psi + sin_phi * sin_theta_sin_psi
    mat_params[2][1] = sin_phi_cos_theta
    mat_params[0][2] = sin_phi_sin_psi + cos_phi * sin_theta_cos_psi
    mat_params[1][2] = -sin_phi_cos_psi + cos_phi * sin_theta_sin_psi
    mat_params[2][2] = cos_phi_cos_theta
    return (mat_params)
#this was an old version of the to matrix complete in which some parameters where blocked to simplify registration
def to_matrix_blocked(vector_params):
    mat_params=np.zeros((3,4))
    mat_params[0][3]=vector_params[0] 
    mat_params[1][3]=vector_params[1] 
    mat_params[2][3]=0
    if vector_params[2] > 1 or vector_params[2] < -1:
        mat_params[0][0]=1 #cos_teta
        mat_params[1][1]=1 #cos_teta
        mat_params[0][1]=0
        mat_params[1][0]=0
        mat_params[0][2]=0
        mat_params[1][2]=0
        mat_params[2][0]=0
        mat_params[2][1]=0
        mat_params[2][2]=1
    else:
        mat_params[0][0]=vector_params[2] #cos_teta
        mat_params[1][1]=vector_params[2] #cos_teta
        mat_params[2][2]= 1
        mat_params[0][1]= -np.sqrt(1-(vector_params[2]**2))
        mat_params[1][0]= -mat_params[0][1]

        mat_params[0][2]=0
        mat_params[1][2]=0
        mat_params[2][0]=0
        mat_params[2][1]=0

    return (mat_params)

def save_data(final_img,list_name,res_path):
    for i in range(len(final_img)):

        b=list_name[i].split('/')
        c=b.pop()
        d=c.split('.')
        cv2.imwrite(res_path+'/'+d[0][0:2]+str(int(d[0][2:5])+1)+'.png', final_img[i])


