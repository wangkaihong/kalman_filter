import numpy as np
import matplotlib.pyplot as plt

class Projector:
    intrinsic_matrix = np.array([[700.,    0.,  320.],
                             [0.,  700.,  240.],
                             [0.,    0.,    1.]])


    def project(self,pts_3d, v, degree, trans):
        rt_pts_3d = self.rotate(pts_3d,v,degree) + trans
        # print(pts_3d)
        # print(rt_pts_3d)
        pts_2d=np.matmul(rt_pts_3d,self.intrinsic_matrix.T) # [8,3] * [3,3]
        # print(pts_2d)
        pts_2d_ret=pts_2d[:,:2]/pts_2d[:,2:]
        # print(pts_2d_ret)
        return pts_2d_ret

    def rotate(self, points, v, degree):
        angle = degree / 180. * np.pi
        v /= np.linalg.norm(v)
        x = v[0]
        y = v[1]
        z = v[2]
        rt = np.array([[np.cos(angle) + (1 - np.cos(angle)) * (x ** 2), (1 - np.cos(angle)) * x * y - np.sin(angle) * z, (1 - np.cos(angle)) * x * z + np.sin(angle) * y],
                       [(1 - np.cos(angle)) * x * y + np.sin(angle) * z, np.cos(angle) + (1 - np.cos(angle)) * (y ** 2), (1 - np.cos(angle)) * y * z + np.sin(angle) * x],
                       [(1 - np.cos(angle)) * z * x - np.sin(angle) * y, (1 - np.cos(angle)) * z * y + np.sin(angle) * x, np.cos(angle) + (1 - np.cos(angle)) * (z ** 2)]])
        # print(rt)
        # print(points)
        return np.matmul(points,rt.T)

def vis(troj, original_troj):
    troj = np.squeeze(troj)
    original_troj = np.squeeze(original_troj)
    p = Projector()

    center = np.array([320, 240])

    vertex1 = p.project(troj, (1, 0, 0), 0, (0, 0, 0))
    vertex2 = p.project(troj, (1, 1, 1), 45, (0, -140000, 200000))
    vertex3 = p.project(troj, (0, 1, 0), 90, (-140000, 0, 200000))
    vertex4 = p.project(troj, (1, -1, 1), 45, (0, 140000, 200000))

    o_vertex1 = p.project(original_troj, (1, 0, 0), 0, (0, 0, 0))
    o_vertex2 = p.project(original_troj, (1, 1, 1), 45, (0, -140000, 200000))
    o_vertex3 = p.project(original_troj, (0, 1, 0), 90, (-140000, 0, 200000))
    o_vertex4 = p.project(original_troj, (1, -1, 1), 45, (0, 140000, 200000))

    d = 10

    colors = ["#" + ("00" + ((hex(i)[2:]).lstrip("0")))[-2:] * 3 for i in
              np.arange(0, 255, int(255 / len(vertex1[:, 0])))]
    colors2 = ["#FF" + ("00" + ((hex(i)[2:]).lstrip("0")))[-2:] * 2 for i in
               np.arange(0, 255, int(255 / len(vertex1[:, 0])))]

    plt.figure()

    plt.subplot(2, 4, 1)
    plt.scatter(vertex1[:, 0], vertex1[:, 1], c=colors)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.subplot(2, 4, 2)
    plt.scatter(vertex2[:, 0], vertex2[:, 1], c=colors)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.subplot(2, 4, 3)
    plt.scatter(vertex3[:, 0], vertex3[:, 1], c=colors)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.subplot(2, 4, 4)
    plt.scatter(vertex4[:, 0], vertex4[:, 1], c=colors)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.subplot(2, 4, 5)
    plt.scatter(vertex1[:, 0], vertex1[:, 1], c=colors)
    plt.scatter(o_vertex1[:, 0], o_vertex1[:, 1], c=colors2)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.subplot(2, 4, 6)
    plt.scatter(vertex2[:, 0], vertex2[:, 1], c=colors)
    plt.scatter(o_vertex2[:, 0], o_vertex2[:, 1], c=colors2)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.subplot(2, 4, 7)
    plt.scatter(vertex3[:, 0], vertex3[:, 1], c=colors)
    plt.scatter(o_vertex3[:, 0], o_vertex3[:, 1], c=colors2)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.subplot(2, 4, 8)
    plt.scatter(vertex4[:, 0], vertex4[:, 1], c=colors)
    plt.scatter(o_vertex4[:, 0], o_vertex4[:, 1], c=colors2)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.show()

def vis_pred(troj, original_troj,pred):
    troj = np.squeeze(troj)
    original_troj = np.squeeze(original_troj)
    pred = np.squeeze(pred)
    p = Projector()

    center = np.array([320, 240])

    vertex1 = p.project(troj, (1, 0, 0), 0, (0, 0, 0))
    vertex2 = p.project(troj, (1, 1, 1), 45, (0, -140000, 200000))
    vertex3 = p.project(troj, (0, 1, 0), 90, (-140000, 0, 200000))
    vertex4 = p.project(troj, (1, -1, 1), 45, (0, 140000, 200000))

    o_vertex1 = p.project(original_troj, (1, 0, 0), 0, (0, 0, 0))
    o_vertex2 = p.project(original_troj, (1, 1, 1), 45, (0, -140000, 200000))
    o_vertex3 = p.project(original_troj, (0, 1, 0), 90, (-140000, 0, 200000))
    o_vertex4 = p.project(original_troj, (1, -1, 1), 45, (0, 140000, 200000))

    p_vertex1 = p.project(pred, (1, 0, 0), 0, (0, 0, 0))
    p_vertex2 = p.project(pred, (1, 1, 1), 45, (0, -140000, 200000))
    p_vertex3 = p.project(pred, (0, 1, 0), 90, (-140000, 0, 200000))
    p_vertex4 = p.project(pred, (1, -1, 1), 45, (0, 140000, 200000))

    d = 10

    colors = ["#" + ("00" + ((hex(i)[2:]).lstrip("0")))[-2:] * 3 for i in
              np.arange(0, 255, int(255 / len(vertex1[:, 0])))]
    colors2 = ["#FF" + ("00" + ((hex(i)[2:]).lstrip("0")))[-2:] * 2 for i in
               np.arange(0, 255, int(255 / len(vertex1[:, 0])))]
    colors3 = ["#" + ("00" + ((hex(i)[2:]).lstrip("0")))[-2:] * 2 +"FF" for i in
               np.arange(0, 255, int(255 / len(vertex1[:, 0])))]

    plt.figure()

    plt.subplot(3, 4, 1)
    plt.scatter(vertex1[:, 0], vertex1[:, 1], c=colors)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.subplot(3, 4, 2)
    plt.scatter(vertex2[:, 0], vertex2[:, 1], c=colors)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.subplot(3, 4, 3)
    plt.scatter(vertex3[:, 0], vertex3[:, 1], c=colors)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.subplot(3, 4, 4)
    plt.scatter(vertex4[:, 0], vertex4[:, 1], c=colors)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.subplot(3, 4, 5)
    plt.scatter(vertex1[:, 0], vertex1[:, 1], c=colors)
    plt.scatter(o_vertex1[:, 0], o_vertex1[:, 1], c=colors2)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.subplot(3, 4, 6)
    plt.scatter(vertex2[:, 0], vertex2[:, 1], c=colors)
    plt.scatter(o_vertex2[:, 0], o_vertex2[:, 1], c=colors2)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.subplot(3, 4, 7)
    plt.scatter(vertex3[:, 0], vertex3[:, 1], c=colors)
    plt.scatter(o_vertex3[:, 0], o_vertex3[:, 1], c=colors2)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.subplot(3, 4, 8)
    plt.scatter(vertex4[:, 0], vertex4[:, 1], c=colors)
    plt.scatter(o_vertex4[:, 0], o_vertex4[:, 1], c=colors2)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.subplot(3, 4, 9)
    plt.scatter(vertex1[:, 0], vertex1[:, 1], c=colors)
    # plt.scatter(o_vertex1[:, 0], o_vertex1[:, 1], c=colors2)
    plt.scatter(p_vertex1[:, 0], p_vertex1[:, 1], c=colors3)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.subplot(3, 4, 10)
    plt.scatter(vertex2[:, 0], vertex2[:, 1], c=colors)
    # plt.scatter(o_vertex2[:, 0], o_vertex2[:, 1], c=colors2)
    plt.scatter(p_vertex2[:, 0], p_vertex2[:, 1], c=colors3)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.subplot(3, 4, 11)
    plt.scatter(vertex3[:, 0], vertex3[:, 1], c=colors)
    # plt.scatter(o_vertex3[:, 0], o_vertex3[:, 1], c=colors2)
    plt.scatter(p_vertex3[:, 0], p_vertex3[:, 1], c=colors3)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.subplot(3, 4, 12)
    plt.scatter(vertex4[:, 0], vertex4[:, 1], c=colors)
    # plt.scatter(o_vertex4[:, 0], o_vertex4[:, 1], c=colors2)
    plt.scatter(p_vertex4[:, 0], p_vertex4[:, 1], c=colors3)
    # plt.xlim((0,640))
    # plt.ylim((0,480))

    plt.show()

class Tracker:

    def __init__(self, dimension= 3):
        self.F = np.array([[1., 1.],[0., 1.]])
        self.H = np.array([1.0, 0.0])
        self.Q = np.ones([2, 2]) * 0.001
        self.R = 1
        self.dimension = dimension

    def gnnsf(self, xk_1, position, H, th=10000000000):
        pos = np.matmul(H, xk_1)
        n = xk_1.shape[0]
        m = position.shape[0]
        p = np.expand_dims(pos, axis=1)
        p = np.repeat(p, m, axis=1)
        mea = np.expand_dims(position, axis=1)
        mea = np.repeat(mea, n, axis=1)
        mea = mea.transpose((1, 0, 2))
        s = p - mea
        s = np.square(s[:, :, 0]) + np.square(s[:, :, 1])
        arg_min = np.argmin(s, axis=1)
        s_min = np.min(s, axis=1)
        min_dis = s_min
        l = np.less(min_dis, th)
        wl = np.where(l == False)[0]
        z = position[arg_min]
        z = np.delete(z, wl, axis=0)
        z_min = np.min(s, axis=0)
        l1 = np.greater(z_min, 625)
        wl1 = np.where(l1 == True)[0]
        new_x = position[wl1, :]
        return z, wl, new_x

    def kalman_filter(self, xk, pk, position, Q, R, H, F):
        n = xk.shape[0]
        _xk_1 = np.matmul(F, xk)
        _pk_1 = np.matmul(np.matmul(F, pk), F.transpose()) + Q

        kk = np.divide(np.matmul(_pk_1, H.transpose()),
                       (np.matmul(np.matmul(H, _pk_1), H.transpose()) + R).reshape(n, 1))
        kk = kk.reshape(n, 2, 1)
        zk_1, wl, new_p = self.gnnsf(_xk_1, position[0], H)
        _xk_1 = np.delete(_xk_1, wl, axis=0)
        _pk_1 = np.delete(_pk_1, wl, axis=0)
        kk = np.delete(kk, wl, axis=0)

        n = _xk_1.shape[0]
        xk_1 = _xk_1 + np.matmul(kk, (zk_1 - np.matmul(H, _xk_1)).reshape(n, 1, self.dimension))
        H_ = np.repeat(np.expand_dims(H, axis=0), n, axis=0)
        H_ = np.reshape(H_, [n, 1, 2])
        I_ = np.repeat(np.expand_dims(np.identity(2), axis=0), n, axis=0)
        pk_1 = np.matmul(I_ - np.matmul(kk, H_), _pk_1)
        return xk_1, pk_1, new_p, wl

    def track(self,troj):
        lines = []
        ret = []
        colors = []
        init_point = np.array(troj[0])
        zero = np.zeros_like(init_point)
        xk = np.concatenate((init_point, zero), axis=1)
        xk = np.reshape(xk, [xk.shape[0], 2, self.dimension])
        pk = np.matmul(self.F, self.F.transpose()) * self.R + self.Q
        pk = np.expand_dims(pk, axis=0)
        pk = np.repeat(pk, xk.shape[0], axis=0)

        for i in range(1,len(troj)):
            xk_1, pk_1, new_p, wl = self.kalman_filter(xk, pk, troj[i], self.Q, self.R, self.H, self.F)
            draw_xk = np.delete(xk, wl, axis=0)
            last_end = [line[-1][-1] for line in lines]

            for i in range(xk_1.shape[0]):
                if (int(draw_xk[i,0,1]),int(draw_xk[i,0,0])) in last_end:
                    ret.append([xk_1[0,0].tolist()])
                    ind = last_end.index((int(draw_xk[i,0,1]),int(draw_xk[i,0,0])))
                    lines[ind].append([(int(draw_xk[i,0,1]),int(draw_xk[i,0,0])),(int(xk_1[i,0,1]),int(xk_1[i,0,0]))])
                else:
                    lines.append([[(int(draw_xk[i,0,1]),int(draw_xk[i,0,0])),(int(xk_1[i,0,1]),int(xk_1[i,0,0])) ]])
                    colors.append((np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)))

            xk = xk_1
            pk = pk_1

        return lines,ret