# from matplotlib import pyplot as plt
import cv2
import numpy as np
from skimage import draw
import math
import copy
import pandas as pd
# from PyQt5.QtWebEngineWidgets import QWebEngineView
# from PyQt5.QtWidgets import QApplication, QMainWindow
# from plotly.graph_objects import Figure, Scatter
import plotly.graph_objects as go
import plotly.express as px
import plotly
import numpy as np
import os

def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    [x, y, w, h] = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    mid = ymin + (ymax - ymin)/2
    global r_start 
    global r_end
    global c_start
    global c_end
    r_start = int(xmin)+1
    r_end = int(xmax)-1
    c_start = int(mid)
    c_end = int(mid)
    print('X start:',xmin)
    print('X end:',xmax)
    print('Y-Mid:',mid)
    return xmin, ymin, xmax, ymax

def draw_boxes(bbox, image, color,crop=False):
    '''
    return image with bounding box
    '''
    left, top, right, bottom = bbox2points(bbox)
    
    
    cv2.rectangle(image, (left, top), (right, bottom), color, 1)
    if crop == True:
        crop_img = image[max(0,top-100):min(bottom+100,image.shape[0]), max(left-100,0):min(right+100,image.shape[1])]
        return crop_img
    
    return image
    


def get_scanline_intensity(image):
    image_copy = image.copy()
    line = np.transpose(np.array(draw.line(r_start, c_start, r_end, c_end)))
    intensity = image_copy.copy()[line[:, 1], line[:, 0]]
    return intensity

def get_start_end_loc(intensity):
    total_luminance = 0
    avg_luminance = 0
    for n,i_n in enumerate(intensity):
        if n!=0:
            avg_luminance = total_luminance/n
        if i_n < 0.85*avg_luminance:
            print('O_l found!')
            break
        total_luminance += i_n
    global o_l
    o_l = r_start+n # xmin + 1 + n
    n1 = n
    print('O_l is:',o_l) 
    total_luminance = 0
    avg_luminance = 0
    for n,i_n in enumerate(intensity[::-1]):
        if n!=0:
            avg_luminance = total_luminance/n
        if i_n < 0.85*avg_luminance:
            print('O_r found!')
            break
        total_luminance += i_n
    global o_r
    o_r = r_end-n # xmax - 1 - n
    n2 = n
    print('O_r is:',o_r)
    intensity = intensity[n1:-n2]
    
    return intensity

def get_ub_uw_var(intensity):
    sorted_intensity = sorted(intensity)
    middle_index = len(sorted_intensity)//2
    ub = np.mean(sorted_intensity[:middle_index])
    uw = np.mean(sorted_intensity[middle_index:])
    var = np.var(sorted_intensity)
    return ub,uw,var

def get_upc_models(ub,uw,intensity):
    m0 = [ub,uw,uw,uw,ub,ub,uw,ub,uw]
    m1 = [ub,uw,uw,ub,ub,uw,uw,ub,uw]
    m2 = [ub,uw,uw,ub,uw,uw,ub,ub,uw]
    m3 = [ub,uw,ub,ub,ub,ub,uw,ub,uw]
    m4 = [ub,uw,ub,uw,uw,uw,ub,ub,uw]
    m5 = [ub,uw,ub,ub,uw,uw,uw,ub,uw]
    m6 = [ub,uw,ub,uw,ub,ub,ub,ub,uw]
    m7 = [ub,uw,ub,ub,ub,uw,ub,ub,uw]
    m8 = [ub,uw,ub,ub,uw,ub,ub,ub,uw]
    m9 = [ub,uw,uw,uw,ub,uw,ub,ub,uw]

    m0_ = [uw,ub,ub,ub,uw,uw,ub,uw,ub]
    m1_ = [uw,ub,ub,uw,uw,ub,ub,uw,ub]
    m2_ = [uw,ub,ub,uw,ub,ub,uw,uw,ub]
    m3_ = [uw,ub,uw,uw,uw,uw,ub,uw,ub]
    m4_ = [uw,ub,uw,ub,ub,ub,uw,uw,ub]
    m5_ = [uw,ub,uw,uw,ub,ub,ub,uw,ub]
    m6_ = [uw,ub,uw,ub,uw,uw,uw,uw,ub]
    m7_ = [uw,ub,uw,uw,uw,ub,uw,uw,ub]
    m8_ = [uw,ub,uw,uw,ub,uw,uw,uw,ub]
    m9_ = [uw,ub,ub,ub,uw,ub,uw,uw,ub]
    
    xm = np.arange(-1,8)
    
    dict_m0 = dict(zip(xm, m0))
    dict_m1 = dict(zip(xm, m1))
    dict_m2 = dict(zip(xm, m2))
    dict_m3 = dict(zip(xm, m3))
    dict_m4 = dict(zip(xm, m4))
    dict_m5 = dict(zip(xm, m5))
    dict_m6 = dict(zip(xm, m6))
    dict_m7 = dict(zip(xm, m7))
    dict_m8 = dict(zip(xm, m8))
    dict_m9 = dict(zip(xm, m9))

    dict_m0_ = dict(zip(xm, m0_))
    dict_m1_ = dict(zip(xm, m1_))
    dict_m2_ = dict(zip(xm, m2_))
    dict_m3_ = dict(zip(xm, m3_))
    dict_m4_ = dict(zip(xm, m4_))
    dict_m5_ = dict(zip(xm, m5_))
    dict_m6_ = dict(zip(xm, m6_))
    dict_m7_ = dict(zip(xm, m7_))
    dict_m8_ = dict(zip(xm, m8_))
    dict_m9_ = dict(zip(xm, m9_))

    model_list1 = [dict_m0,dict_m1,dict_m2,dict_m3,dict_m4,dict_m5,dict_m6,dict_m7,dict_m8,dict_m9]
    model_list2 = [dict_m0_,dict_m1_,dict_m2_,dict_m3_,dict_m4_,dict_m5_,dict_m6_,dict_m7_,dict_m8_,dict_m9_]
    
    w = len(intensity)/95
    xm_ = ((xm)*w)+3*w

    global model_fig
    global intensity_fig
    
    model_fig = go.Figure()
    intensity_fig = go.Figure()
    
    model_fig.add_trace(go.Scatter(x=xm, y=m0, name="M0",visible='legendonly',
                    line_shape='hv'))
    
    model_fig.add_trace(go.Scatter(x=xm, y=m1, name="M1",visible='legendonly',
                        line_shape='hv'))
    model_fig.add_trace(go.Scatter(x=xm, y=m2, name="M2",
                        line_shape='hv'))
    model_fig.add_trace(go.Scatter(x=xm, y=m3, name="M3",visible='legendonly',
                        line_shape='hv'))
    model_fig.add_trace(go.Scatter(x=xm, y=m4, name="M4",visible='legendonly',
                        line_shape='hv'))
    model_fig.add_trace(go.Scatter(x=xm, y=m5, name="M5",visible='legendonly',
                        line_shape='hv'))
    model_fig.add_trace(go.Scatter(x=xm, y=m6, name="M6",visible='legendonly',
                        line_shape='hv'))
    model_fig.add_trace(go.Scatter(x=xm, y=m7, name="M7",visible='legendonly',
                        line_shape='hv'))
    model_fig.add_trace(go.Scatter(x=xm, y=m8, name="M8",visible='legendonly',
                        line_shape='hv'))
    model_fig.add_trace(go.Scatter(x=xm, y=m9, name="M9",visible='legendonly',
                        line_shape='hv'))
    
    intensity_fig.add_trace(go.Scatter(x=xm_, y=m0, name="M0",visible='legendonly',
                    line_shape='hv'))
    
    intensity_fig.add_trace(go.Scatter(x=xm_, y=m1, name="M1",visible='legendonly',
                        line_shape='hv'))
    intensity_fig.add_trace(go.Scatter(x=xm_, y=m2, name="M2",
                        line_shape='hv'))
    intensity_fig.add_trace(go.Scatter(x=xm_, y=m3, name="M3",visible='legendonly',
                        line_shape='hv'))
    intensity_fig.add_trace(go.Scatter(x=xm_, y=m4, name="M4",visible='legendonly',
                        line_shape='hv'))
    intensity_fig.add_trace(go.Scatter(x=xm_, y=m5, name="M5",visible='legendonly',
                        line_shape='hv'))
    intensity_fig.add_trace(go.Scatter(x=xm_, y=m6, name="M6",visible='legendonly',
                        line_shape='hv'))
    intensity_fig.add_trace(go.Scatter(x=xm_, y=m7, name="M7",visible='legendonly',
                        line_shape='hv'))
    intensity_fig.add_trace(go.Scatter(x=xm_, y=m8, name="M8",visible='legendonly',
                        line_shape='hv'))
    intensity_fig.add_trace(go.Scatter(x=xm_, y=m9, name="M9",visible='legendonly',
                        line_shape='hv'))

    return model_list1,model_list2

def decode_upc(ub,uw,var,intensity,model_list1,model_list2):
    x = np.arange(len(intensity))
    dict_I = dict(zip(x,intensity))
    o_start = 0  
    result_x = np.arange(1,13)
    result_y = [[100]*10]*12
    result_dict = dict(zip(result_x,result_y))
    d_w = 0
    d_o = 0
    w = len(intensity)/95
    if w - (math.floor(w))<= 0.5:
        do = math.floor(w)
    else:
        do = math.floor(w)+1

    for d_o in np.arange(-do, do+1, 1):
        for d_w in np.arange(-0.025,0.025+0.01,0.01):
            # first segment analyze
            try:
                detected = []
                for j in range(1,13):
                    temp_res = copy.deepcopy(result_dict[j])
                    if j >= 7:
                        o = o_start+8*(w+d_w)+7*(w+d_w)*(j-1)
                        model_list = model_list2
                    else:
                        o = o_start+3*(w+d_w)+7*(w+d_w)*(j-1)
                        model_list = model_list1
                    D_result = []
                    seg_start = math.ceil(o-(w+d_w))
                    seg_end = math.floor(o+8*(w+d_w))
                    for k,dict_m in enumerate(model_list):
                        D_list = []
                        for n in range(seg_start, seg_end+1):
                            I_n = dict_I[n]
                            m_x = math.floor((n-o)/w)
                            m_y = dict_m[m_x]
                            if m_y == ub:
                                D = ((max(0,I_n-ub))**2)/(2*var**2)
                                D_list.append(D)
                            elif m_y == uw:
                                D = ((min(0,I_n-uw))**2)/(2*var**2)
                                D_list.append(D)
                            else:
                                print('ERROR!!!!!!')
                        D_result.append(sum(D_list))
                    k = D_result.index(min(D_result))
                    if min(D_result) < temp_res[k]:
                        temp_res[k] = min(D_result)
                        result_dict[j] = temp_res
                    temp_res = []
                    detected.append(D_result.index(min(D_result)))
            except:
                continue
    final_res = []
    for res in result_dict.values():
        k = res.index(min(res))
        final_res.append(k)
    print('Final Result:',final_res)
    return final_res

def get_code128_models(ub,uw,intensity):
    global model_fig
    global intensity_fig
    
    model_fig = go.Figure()
    intensity_fig = go.Figure()
    
    
    model_df = pd.read_csv('dependencies/code128_model.csv')
    w = len(intensity)/90
    xm = np.arange(-1,12)
    xm_ = ((xm)*w)+11*w
    model_name = []
    model_list = []
    for i,row in model_df.iterrows():
        model_name.append(row['128C'])
        m = []
        for item in str(row['图案']):
            if item == '0':
                m.append(uw)
            if item == '1':
                m.append(ub)
        m.append(ub)
        m.insert(0,uw)
        dict_m = dict(zip(xm, m))
        if i == 0:
            model_fig.add_trace(go.Scatter(x=xm, y=m, name=row['128C'],
                        line_shape='hv'))
            intensity_fig.add_trace(go.Scatter(x=xm_, y=m, name=row['128C'],
                        line_shape='hv'))
        else:
            model_fig.add_trace(go.Scatter(x=xm, y=m, name=row['128C'],visible='legendonly',
                    line_shape='hv'))
            intensity_fig.add_trace(go.Scatter(x=xm_, y=m, name=row['128C'],visible='legendonly',
            line_shape='hv'))
        model_list.append(dict_m)
    len(model_list)
    return model_name,model_list

def decode_code128(ub,uw,var,intensity,model_name,model_list):
    result_x = np.arange(1,7)
    result_y = [[100]*109]*6
    result_dict = dict(zip(result_x,result_y))
    d_w = 0
    d_o = 0
    o_start = 0
    x = np.arange(len(intensity))
    dict_I = dict(zip(x,intensity))
    o_start = 0  
    w = len(intensity)/90
    if w - (math.floor(w))<= 0.5:
        do = math.floor(w)
    else:
        do = math.floor(w)+1
    for d_o in np.arange(-do, do+1, 1):
        for d_w in np.arange(-0.025,0.025+0.01,0.005):
            # first segment analyze
            try:
                detected = []
                for j in range(1,7):
                    temp_res = copy.deepcopy(result_dict[j])
                    o = o_start+11*(w+d_w)+11*(w+d_w)*(j-1)
                    D_result = []
                    seg_start = math.ceil(o-(w+d_w))
                    seg_end = math.floor(o+12*(w+d_w))
                    for k,dict_m in enumerate(model_list[:100]):
                        D_list = []
                        for n in range(seg_start, seg_end+1):
                            I_n = dict_I[n]
                            m_x = math.floor((n-o)/w)
                            m_y = dict_m[m_x]
                            if m_y == ub:
                                D = ((max(0,I_n-ub))**2)/(2*var**2)
                                D_list.append(D)
                            elif m_y == uw:
                                D = ((min(0,I_n-uw))**2)/(2*var**2)
                                D_list.append(D)
                            else:
                                print('ERROR!!!!!!')
                        D_result.append(sum(D_list))
                    k = D_result.index(min(D_result))
                    if min(D_result) < temp_res[k]:
                        temp_res[k] = min(D_result)
                        result_dict[j] = temp_res
                    temp_res = []
                    detected.append(D_result.index(min(D_result)))
            except:
                continue
    final_res = []
    for res in result_dict.values():
        k = res.index(min(res))
        final_res.append(model_name[k])
    print('Final Result:',final_res)
    return final_res

def get_code39_models(ub,uw,intensity):
    global model_fig
    global intensity_fig
    
    model_fig = go.Figure()
    intensity_fig = go.Figure()
    
    df = pd.read_csv('dependencies/code39_model.csv') 
    xm = np.arange(-1,16)
    w = len(intensity)/90
    xm_ = ((xm)*w)+15*w
    model_name = []
    model_list = []
    for i,row in df[1:].iterrows():
        model_name.append(row[0])
        temp = [uw]
        for item in row[1]:
            if item == 'N':
                if temp[-1]==uw:
                    temp.append(ub)
                    continue
                if temp[-1]==ub:
                    temp.append(uw)
                    continue
            if item == 'W':
                if temp[-1]==uw:
                    temp.append(ub)
                    temp.append(ub)
                    temp.append(ub)
                    continue
                if temp[-1]==ub:
                    temp.append(uw) 
                    temp.append(uw) 
                    temp.append(uw) 
                    continue
        temp.append(uw)
        dict_m = dict(zip(xm, temp))
        if i == 1:
            model_fig.add_trace(go.Scatter(x=xm, y=temp, name=row[0],
                        line_shape='hv'))
            intensity_fig.add_trace(go.Scatter(x=xm_, y=temp, name=row[0],
                        line_shape='hv'))
        else:
            model_fig.add_trace(go.Scatter(x=xm, y=temp, name=row[0],visible='legendonly',
                    line_shape='hv'))
            intensity_fig.add_trace(go.Scatter(x=xm_, y=temp, name=row[0],visible='legendonly',
                    line_shape='hv'))
        model_list.append(dict_m) 
    return model_name,model_list

def decode_code39(ub,uw,var,intensity,model_name,model_list):
    x = np.arange(len(intensity))
    dict_I = dict(zip(x,intensity))
    result_x = np.arange(1,5)
    result_y = [[100]*44]*4
    result_dict = dict(zip(result_x,result_y))
    d_w = 0
    d_o = 0
    o_start = 0
    w = len(intensity)/90
    if w - (math.floor(w))<= 0.5:
        do = math.floor(w)
    else:
        do = math.floor(w)+1
    for d_o in np.arange(-do, do+1, 1):
        for d_w in np.arange(-0.05,0.05+0.01,0.006):
            # first segment analyze
            try:
                detected = []
                for j in range(1,5):
                    temp_res = copy.deepcopy(result_dict[j])
                    o = o_start+15*(w+d_w)+15*(w+d_w)*(j-1)
                    D_result = []
                    seg_start = math.ceil(o-(w+d_w))
                    seg_end = math.floor(o+16*(w+d_w))
                    for k,dict_m in enumerate(model_list):
                        D_list = []
                        for n in range(seg_start, seg_end+1):
                            I_n = dict_I[n]
                            m_x = math.floor((n-o)/w)
                            m_y = dict_m[m_x]
                            if m_y == ub:
                                D = ((max(0,I_n-ub))**2)/(2*var**2)
                                D_list.append(D)
                            elif m_y == uw:
                                D = ((min(0,I_n-uw))**2)/(2*var**2)
                                D_list.append(D)
                            else:
                                print('ERROR!!!!!!')
                        D_result.append(sum(D_list))
                    k = D_result.index(min(D_result))
                    if min(D_result) < temp_res[k]:
                        temp_res[k] = min(D_result)
                        result_dict[j] = temp_res
                    temp_res = []
                    detected.append(D_result.index(min(D_result)))
            except:
                continue
    final_res = []
    for res in result_dict.values():
        k = res.index(min(res))
        final_res.append(model_name[k])
    print('Final Result:',final_res)
    return final_res

def decode_image(imagename):
    print('DECODING ......')
    global o_l
    global o_r
    global c_start
    global c_end
    global intensity_fig
    global color_locate
    global draw_circle_image
    color = (0,0,255)

    if imagename in [r'dependencies/images/070470409665.jpg',r'dependencies/images/689076338486.jpg',r'dependencies/images/014149929962.bmp']:
        print('UPC-A')
        if imagename == r'dependencies/images/070470409665.jpg':
            bbox = [910,850,183,140]
            color_image = cv2.imread(imagename)
            color_image = cv2.resize(color_image,(color_image.shape[1]*2,color_image.shape[0]*2),
                                    interpolation=cv2.INTER_CUBIC)

            color_locate = draw_boxes(bbox,color_image,color,True)
            
            image = cv2.imread(imagename,0)
            image = cv2.resize(image,(image.shape[1]*2,image.shape[0]*2),
                                    interpolation=cv2.INTER_CUBIC)
        if imagename == r'dependencies/images/689076338486.jpg':
            bbox = [1420,950,275,140]
            color_image = cv2.imread(imagename)
            color_image = cv2.resize(color_image,(color_image.shape[1]*2,color_image.shape[0]*2),
                                    interpolation=cv2.INTER_CUBIC)

            color_locate = draw_boxes(bbox,color_image,color,True)
            
            image = cv2.imread(imagename,0)
            image = cv2.resize(image,(image.shape[1]*2,image.shape[0]*2),
                                    interpolation=cv2.INTER_CUBIC)

        if imagename == r'dependencies/images/014149929962.bmp':
            bbox = [277,220,231,67]
            color_image = cv2.imread(imagename)
        
            color_locate = draw_boxes(bbox,color_image,color,True)
            image = cv2.imread(imagename,0)
        
        located = draw_boxes(bbox,image,color)
        intensity = get_scanline_intensity(located)
        
        draw_line_image = cv2.line(located.copy(), (r_start, c_start), (r_end, c_end), (0, 0, 0), 2)
        intensity = get_start_end_loc(intensity)
        
        
        
        draw_circle_image =cv2.circle(draw_line_image, (o_l,c_start), 2, (255, 0, 0), 1)
        draw_circle_image = cv2.circle(draw_circle_image, (o_r,c_end), 2, (255, 0, 0), 1)   
        draw_circle_image = draw_boxes(bbox,draw_circle_image,color,True)
        
        ub,uw,var = get_ub_uw_var(intensity)

        model_list1,model_list2 = get_upc_models(ub,uw,intensity)
        x = np.arange(len(intensity))
        intensity_fig.add_trace(go.Scatter(x=x, y=intensity,
                    mode='lines+markers',
                    name='Intensity Profile'))
        result = decode_upc(ub,uw,var,intensity,model_list1,model_list2)
        
        return result
    
    if imagename in [r'dependencies/images/Code128 5mil 10chars 18cm.jpg',r'dependencies/images/Code128 5mil 10chars 36cm.jpg',r'dependencies/images/code128 5mil 10chars.jpg']:
        print('CODE128')
        if imagename == r'dependencies/images/Code128 5mil 10chars 18cm.jpg':
            bbox = [1200,761,400,249]
            color_image = cv2.imread(imagename)
            color_bbox = [600,380,200,124]

            color_locate = draw_boxes(color_bbox,color_image,color,True)
            
            image = cv2.imread(imagename,0)
            image = cv2.resize(image,(image.shape[1]*2,image.shape[0]*2),
                                    interpolation=cv2.INTER_CUBIC)
            
            located = draw_boxes(bbox,image,color)
            intensity = get_scanline_intensity(located)
            
            draw_line_image = cv2.line(color_image.copy(), (r_start//2, c_start//2), (r_end//2, c_end//2), (0, 0, 0), 2)
            intensity = get_start_end_loc(intensity)
            
            
            draw_circle_image =cv2.circle(draw_line_image, (o_l//2,c_start//2), 2, (255, 0, 0), 1)
            draw_circle_image = cv2.circle(draw_circle_image, (o_r//2,c_end//2), 2, (255, 0, 0), 1)   
            draw_circle_image = draw_boxes(color_bbox,draw_circle_image,color,True)

        if imagename == r'dependencies/images/Code128 5mil 10chars 36cm.jpg':
            bbox = [1180,800,150,323]
            color_image = cv2.imread(imagename)
            color_image = cv2.resize(color_image,(color_image.shape[1]*2,color_image.shape[0]*2),
                                    interpolation=cv2.INTER_CUBIC)
            color_locate = draw_boxes(bbox,color_image,color,True)
            
            image = cv2.imread(imagename,0)
            image = cv2.resize(image,(image.shape[1]*2,image.shape[0]*2),
                                    interpolation=cv2.INTER_CUBIC)
            located = draw_boxes(bbox,image,color)
            intensity = get_scanline_intensity(located)
            
            draw_line_image = cv2.line(located.copy(), (r_start, c_start), (r_end, c_end), (0, 0, 0), 2)
            intensity = get_start_end_loc(intensity)
            
            
            
            draw_circle_image =cv2.circle(draw_line_image, (o_l,c_start), 2, (255, 0, 0), 1)
            draw_circle_image = cv2.circle(draw_circle_image, (o_r,c_end), 2, (255, 0, 0), 1)   
            draw_circle_image = draw_boxes(bbox,draw_circle_image,color,True)

        if imagename == r'dependencies/images/code128 5mil 10chars.jpg':
            bbox = [300,400,500,590]
            color_bbox = [150,200,250,295]
            color_image = cv2.imread(imagename)
            color_image = cv2.resize(color_image,(color_image.shape[1]//2,color_image.shape[0]//2),
                                    interpolation=cv2.INTER_AREA)
            color_locate = draw_boxes(color_bbox,color_image,color,True)
            image = cv2.imread(imagename,0)
            
            located = draw_boxes(bbox,image,color)
            intensity = get_scanline_intensity(located)
            
            draw_line_image = cv2.line(color_image.copy(), (r_start//2, c_start//2), (r_end//2, c_end//2), (0, 0, 0), 2)
            intensity = get_start_end_loc(intensity)

            draw_circle_image =cv2.circle(draw_line_image, (o_l//2,c_start//2), 2, (255, 0, 0), 1)
            draw_circle_image = cv2.circle(draw_circle_image, (o_r//2,c_end//2), 2, (255, 0, 0), 1)   
            draw_circle_image = draw_boxes(color_bbox,draw_circle_image,color,True)
      
        ub,uw,var = get_ub_uw_var(intensity)

        model_name,model_list = get_code128_models(ub,uw,intensity)
        x = np.arange(len(intensity))
        intensity_fig.add_trace(go.Scatter(x=x, y=intensity,
                    mode='lines+markers',
                    name='Intensity Profile'))
        result = decode_code128(ub,uw,var,intensity,model_name,model_list)
        
        return result
    

    if imagename in [r'dependencies/images/code39 5mil 37cm.jpg',r'dependencies/images/code39 5mil 18cm.jpg',r'dependencies/images/code39 5mil 3chars.jpg']:
        print('CODE39')
        if imagename == r'dependencies/images/code39 5mil 37cm.jpg':
            bbox = [1198,772,200,241]
            color_image = cv2.imread(imagename)
            color_image = cv2.resize(color_image,(color_image.shape[1]*2,color_image.shape[0]*2),
                                    interpolation=cv2.INTER_CUBIC)
            color_locate = draw_boxes(bbox,color_image,color,True)
            
            image = cv2.imread(imagename,0)
            image = cv2.resize(image,(image.shape[1]*2,image.shape[0]*2),
                                    interpolation=cv2.INTER_CUBIC)
            located = draw_boxes(bbox,image,color)
            intensity = get_scanline_intensity(located)
            
            draw_line_image = cv2.line(located.copy(), (r_start, c_start), (r_end, c_end), (0, 0, 0), 2)
            intensity = get_start_end_loc(intensity)

            draw_circle_image =cv2.circle(draw_line_image, (o_l,c_start), 2, (255, 0, 0), 1)
            draw_circle_image = cv2.circle(draw_circle_image, (o_r,c_end), 2, (255, 0, 0), 1)   
            draw_circle_image = draw_boxes(bbox,draw_circle_image,color,True)
        
        if imagename == r'dependencies/images/code39 5mil 18cm.jpg':
            bbox = [558,372,132,241]
            color_image = cv2.imread(imagename)
            color_locate = draw_boxes(bbox,color_image,color,True)
            
            image = cv2.imread(imagename,0)
            located = draw_boxes(bbox,image,color)
            intensity = get_scanline_intensity(located)
            
            draw_line_image = cv2.line(located.copy(), (r_start, c_start), (r_end, c_end), (0, 0, 0), 2)
            intensity = get_start_end_loc(intensity)
            
            
            
            draw_circle_image =cv2.circle(draw_line_image, (o_l,c_start), 2, (255, 0, 0), 1)
            draw_circle_image = cv2.circle(draw_circle_image, (o_r,c_end), 2, (255, 0, 0), 1)   
            draw_circle_image = draw_boxes(bbox,draw_circle_image,color,True)

        if imagename == r'dependencies/images/code39 5mil 3chars.jpg':
            bbox = [653,257,576,514]
            color_bbox = [326,128,288,257]
            color_image = cv2.imread(imagename)
            color_image = cv2.resize(color_image,(color_image.shape[1]//2,color_image.shape[0]//2),
                                    interpolation=cv2.INTER_AREA)

            color_locate = draw_boxes(color_bbox,color_image,color,True)
            image = cv2.imread(imagename,0)
            located = draw_boxes(bbox,image,color)
            intensity = get_scanline_intensity(located)
            
            draw_line_image = cv2.line(color_image.copy(), (r_start//2, c_start//2), (r_end//2, c_end//2), (0, 0, 0), 2)
            
            intensity = get_start_end_loc(intensity) 

            draw_circle_image =cv2.circle(draw_line_image, (o_l//2,c_start//2), 2, (255, 0, 0), 1)
            draw_circle_image = cv2.circle(draw_circle_image, (o_r//2,c_end//2), 2, (255, 0, 0), 1)   
            draw_circle_image = draw_boxes(color_bbox,draw_circle_image,color,True)

        
        
        ub,uw,var = get_ub_uw_var(intensity)

        model_name,model_list = get_code39_models(ub,uw,intensity)
        x = np.arange(len(intensity))
        intensity_fig.add_trace(go.Scatter(x=x, y=intensity,
                    mode='lines+markers',
                    name='Intensity Profile'))
        result = decode_code39(ub,uw,var,intensity,model_name,model_list)
        
        return ['*']+result+['*']
   


def get_html(imagename,result):
    global color_locate
    global draw_circle_image
    global model_fig
    global intensity_fig
    
    locate_fig = px.imshow(color_locate)
    locate_fig.update_layout(
        title="Barcode Localization",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    scanline_fig = px.imshow(draw_circle_image,color_continuous_scale='gray')
    scanline_fig.update_layout(
        title="Scanline Localization",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )

    model_fig.update_layout(
        title="Barcode Decode Models",
        legend_title="Model Name",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )

    intensity_fig.update_layout(
        title="Example of matching intensity with one model",
        xaxis_title="Location(pixel)",
        yaxis_title="Value",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )

    r2 = [2,1,2,2]
    delta_o = 1
    delta_w = 0.5
    do = np.arange(-delta_o,delta_o+1)
    polygon_fig = go.Figure(layout_yaxis_range=[-delta_w,delta_w])
    colors = [['rgba(31, 119, 180, 1)','rgba(31, 119, 180, 0.15)'],
            ['rgba(255, 127, 14, 1)','rgba(255, 127, 14, 0.15)'],
            ['rgba(44, 160, 44, 1)','rgba(44, 160, 44, 0.15)'],
            ['rgba(214, 39, 40, 1)','rgba(214, 39, 40, 0.15)'],
            ['rgba(148, 103, 189, 1)','rgba(148, 103, 189, 0.15)'],
            ]*10
    count = 0
    for i in range(0,4):
        for q in np.arange(-4,5):
            r_sum = sum(r2[:i+1])
            dw = (q-do)/r_sum
            polygon_fig.add_trace(go.Scatter(x=do, y=dw, name="boundary %s = %s"%(i+1,q),
                                fill='tonexty',
                                    line=dict(color=colors[count][0]),
                                    fillcolor=colors[count][1],
                                    line_shape='linear'))
            count += 1
    polygon_fig.update_layout(plot_bgcolor='white',
        title="An example space of (dw,do) broken into polygons",
        xaxis_title="do",
        yaxis_title="dw",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        ))
    
    # we create html code of the figure
    html = '''<html><style>
                    body {
                    background-color: #E6E6FA;
                    }
                    h1 {text-align: center;}
                    </style><body>'''
    html += '<h1 style="font-size:250%;color: #663399;">Barcode Decoding</h1>'
    html += '<h2 style="font-size:170%;color: #663399;">“'+ os.path.split(imagename)[1] + '”Decoding Result: ' +str(result)+ '</h2>'
    html += '<h2 style="font-size:170%;color: #663399;">1. Localization</h2>'
    plotly_fig_1 = plotly.offline.plot(locate_fig, output_type='div', include_plotlyjs='cdn')
    html += plotly_fig_1
    html += '<h2 style="font-size:170%;color: #663399;">2. Start/End points localization</h2>'

    plotly_fig_2 = plotly.offline.plot(scanline_fig, output_type='div', include_plotlyjs='cdn')
    html += plotly_fig_2
    html += '<h2 style="font-size:170%;color: #663399;">3. Loading templates</h2>'
   
    plotly_fig_3 = plotly.offline.plot(model_fig, output_type='div', include_plotlyjs='cdn')
    html += plotly_fig_3
    html += '<h2 style="font-size:170%;color: #663399;">4. Extract intensity profile，match with templates</h2>'
     
    plotly_fig_4 = plotly.offline.plot(intensity_fig, output_type='div', include_plotlyjs='cdn')
    html += plotly_fig_4
    html += '<h2 style="font-size:170%;color: #663399;">5. Move and stretch template, and choose sequence that maximized likelihood</h2>'
     
    plotly_fig_5 = plotly.offline.plot(polygon_fig, output_type='div', include_plotlyjs='cdn')
    html += plotly_fig_5
    html += '<h2 style="font-size:170%;color: #663399;">Conclusion</h2>'
    'matching with all templates one by one, which may lead to less efficient decoding if there are too many templates. For upc-a decoding implemented in the paper, there are only 20 kinds of templates in total, and the matching efficiency is high.'
    html += '<p style="font-size:160%;color: #663399;">（1）The algorithm is based on <b>matching with all templates one by one， which may lead to less efficient decoding if there are too many templates.</b>\n For upc-a decoding, there are only 20 kinds of templates in total, so it is more efficient. But for code128 and code39, there are over 100 templates, the algorithm performs slowly if the coding is completely random.</p>'
    html += '<p style="font-size:160%;color: #663399;">（2）The prerequisite of this algorithm is that <b>width of the narrowest black/white bar is known.</b>\n Since the encoding length of upc-a code is fixed, the minimum unit width can be obtained by knowing the total length of the barcode. However, for indefinite length code like code128 and code39, if the code length is unknown, the algorithm will fail. In this demo, we give this prior knowledge as our application scenario\'s code has fixed length. But for the use case in other applications, how to solve the uncertainty of base width will be the first thing to think about.'
    html += '<p style="font-size:160%;color: #663399;">（3）The <b>decoding limit of this algorithm is that the minimum unit width is greater than 1 pixel point.</b> In sample long-distance codes of code128 and code39, the minimum unit width is about 0.6 pixels, which is beyond the decoding range. \n By resizing the image by 2, the code39 can recognize all of them and the code128 can recognize 3/5 for the code from a long distance. The performance of this specific algorithm in recognizing more far-away and more complicated codes needs further evaluation.'
    html += '</body></html>'
    return html

