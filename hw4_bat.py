import cv2 as cv
import numpy as np
import random

#preprocess an image
def img_process(i,back_red):
    back = np.int32(back_red)
    name_pre = 'FalseColor/CS585Bats-FalseColor_frame000000'
    filename = name_pre + '%d.ppm' % i
    img = cv.imread(filename)
    img = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    img_r = np.copy(img)
    img = img[:, :, 2]
    img = np.int32(img)
    img = img - back
    img = np.greater(img, 0).astype(np.uint8) * np.uint8(img)
    img = np.greater(img, 50).astype(np.uint8) * 255
    img[410:482, :24] = 0
    img[460:480, 495:] = 0
    kernel = cv.getStructuringElement(cv.MORPH_DILATE, (3, 3))
    img = cv.dilate(img, kernel, iterations=4)
    img = cv.erode(img, kernel, iterations=1)
    return img_r,img

#object tracking
def object_tracking():
    Q = np.ones([2,2]) * 0.1
    R = 10
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([1.0, 0.0])
    back_red = red_analyse()
    while True:
        lines = []
        colors = []
        for i in range(750,901):
            img_r, img = img_process(i,back_red)
            mea = localization(img)
            #initialization
            if i == 750:
                zero = np.zeros_like(mea)
                xk = np.concatenate((mea,zero),axis=1)
                xk = np.reshape(xk,[xk.shape[0],2,2])
                pk = np.matmul(A,A.transpose()) * R + Q
                pk = np.expand_dims(pk,axis=0)
                pk = np.repeat(pk,xk.shape[0],axis=0)
                continue
            #kalman filter
            xk_1, pk_1,new_p,wl = kalman_filter(xk,pk,mea,Q,R,H,A)
            #draw lines on image
            draw_xk = np.delete(xk, wl, axis=0)
            last_end = [line[-1][-1] for line in lines]
            for i in range(xk_1.shape[0]):
                if (int(draw_xk[i,0,1]),int(draw_xk[i,0,0])) in last_end:
                    ind = last_end.index((int(draw_xk[i,0,1]),int(draw_xk[i,0,0])))
                    lines[ind].append([(int(draw_xk[i,0,1]),int(draw_xk[i,0,0])),(int(xk_1[i,0,1]),int(xk_1[i,0,0]))])
                else:
                    lines.append([[(int(draw_xk[i,0,1]),int(draw_xk[i,0,0])),(int(xk_1[i,0,1]),int(xk_1[i,0,0])) ]])
                    colors.append((random.randint(0,255),random.randint(0,255),random.randint(0,255)))
            for i in range(len(lines)):
                for j in range(len(lines[i])):
                    cv.line(img_r,lines[i][j][0],lines[i][j][1],colors[i])
            #state transfer
            xk = xk_1
            pk = pk_1
            new_xk, new_pk = new_object(new_p,A,R,Q)
            if new_xk is not None:
                xk = np.concatenate((xk,new_xk),axis=0)
                pk = np.concatenate((pk,new_pk),axis=0)
            cv.imshow('111',img)
            cv.imshow('Bat Tracking',img_r)
            cv.waitKey(100)

#sequential labeling
def localization(seg_mat):
    ret,labels = cv.connectedComponents(seg_mat)
    ret_mat = np.zeros([ret-1,2])
    for i in range(1,ret):
        x,y = int(np.mean(np.where(labels==i)[0])),int(np.mean(np.where(labels==i)[1]))
        ret_mat[i-1,0] = x
        ret_mat[i-1,1] = y
    return ret_mat

#kalman filter
def kalman_filter(xk,pk,can_mea,Q,R,H,A):
    n = xk.shape[0]
    _xk_1 = np.matmul(A,xk)
    _pk_1 = np.matmul(np.matmul(A,pk),A.transpose()) + Q
    kk = np.divide(np.matmul(_pk_1,H.transpose()),(np.matmul(np.matmul(H,_pk_1),H.transpose())+R).reshape(n,1))
    kk = kk.reshape(n,2,1)
    zk_1,wl,new_p = gnnsf(_xk_1,can_mea,H)
    _xk_1 = np.delete(_xk_1,wl,axis=0)
    _pk_1 = np.delete(_pk_1,wl,axis=0)
    kk = np.delete(kk,wl,axis=0)

    n = _xk_1.shape[0]
    xk_1 = _xk_1 + np.matmul(kk,(zk_1-np.matmul(H,_xk_1)).reshape(n,1,2))
    H_ = np.repeat(np.expand_dims(H,axis=0),n,axis=0)
    H_ = np.reshape(H_,[n,1,2])
    I_ = np.repeat(np.expand_dims(np.identity(2),axis=0),n,axis=0)
    pk_1 = np.matmul(I_-np.matmul(kk,H_),_pk_1)
    return xk_1,pk_1,new_p,wl

#data association
def gnnsf(xk_1,can_mea,H,th=625):
    pos = np.matmul(H, xk_1)
    n = xk_1.shape[0]
    m = can_mea.shape[0]
    p = np.expand_dims(pos, axis=1)
    p = np.repeat(p, m, axis=1)
    mea = np.expand_dims(can_mea, axis=1)
    mea = np.repeat(mea, n, axis=1)
    mea = mea.transpose((1, 0, 2))
    s = p - mea
    s = np.square(s[:, :, 0]) + np.square(s[:, :, 1])
    arg_min = np.argmin(s, axis=1)
    s_min = np.min(s, axis=1)
    min_dis = s_min
    l = np.less(min_dis,th)
    wl = np.where(l==False)[0]
    z = can_mea[arg_min]
    z = np.delete(z,wl,axis=0)
    z_min = np.min(s,axis=0)
    l1 = np.greater(z_min,625)
    wl1 = np.where(l1==True)[0]
    new_x = can_mea[wl1,:]
    return z, wl, new_x

#get new objects states
def new_object(new_mea,A,R,Q):
    new_ob = new_mea
    new_zero = np.zeros_like(new_ob)
    new_x = np.concatenate((new_ob, new_zero), axis=1)
    new_x = np.reshape(new_x, [new_x.shape[0], 2, 2])
    new_pk = np.matmul(A, A.transpose()) * R + Q
    new_pk = np.expand_dims(new_pk, axis=0)
    new_pk = np.repeat(new_pk, new_x.shape[0], axis=0)
    return new_x, new_pk


#get background brightness
def red_analyse():
    name_pre = 'FalseColor/CS585Bats-FalseColor_frame000000'
    red_mean = []
    w, h = 1024, 1024
    for i in range(750, 901):
        filename = name_pre + '%d.ppm' % i
        img = cv.imread(filename)
        img = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        img_r = img[:, :, 2]
        red = np.sum(img_r, axis=1) / img_r.shape[1]
        red_mean.append(red)
    Y = np.mean(red_mean,axis=0)
    ret_Y = np.reshape(Y,[Y.shape[0],1])
    ret_Y = np.repeat(ret_Y,w//2,axis=1)
    return ret_Y



if __name__ == '__main__':
    object_tracking()






























































