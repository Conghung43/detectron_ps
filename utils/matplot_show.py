import numpy as np
import matplotlib.pyplot as plt

class data_show_3d():
    def __init__(self, points_x):
        points = np.array(points_x)
        x_flat, y_flat, z_flat = points[:,0], points[:,1], points[:,2]
        # xc_flat, yc_flat, zc_flat = clustering_center_points[:,0], clustering_center_points[:,1], clustering_center_points[:,2]
        fig = plt.figure()
        self.ax = plt.axes(projection="3d")
        # self.ax.scatter3D(x_flat, y_flat, z_flat, c=labels.astype(float), edgecolor='k')
        self.ax.scatter3D(x_flat, y_flat, z_flat )
        # self.ax.scatter3D(xc_flat, yc_flat, zc_flat)

        # points = np.array(points_y)
        # x_flat, y_flat, z_flat = points[:,0], points[:,1], points[:,2]
        # self.ax.scatter3D(x_flat, y_flat, z_flat)

        plt.show()

class data_show_3d_single():
    def __init__(self, points):
        fig = plt.figure()
        self.ax = plt.axes(projection="3d")
        points = np.array(points)
        x_flat, y_flat, z_flat = points[:,0], points[:,1], points[:,2]
        self.ax.scatter3D(x_flat, y_flat, z_flat)

        plt.show()

class data_show_2d():
    def __init__(self, points):
        points = np.array(points)
        x_flat, y_flat = points[:,0], points[:,1]
        self.plt = plt
        fig = self.plt.figure()
        self.ax = self.plt
        self.ax.scatter(x_flat, y_flat)
        plt.show()

# points = [([-178.7028673 ,    2.01738673,  590.00006104]),([-161.03617998,    1.28449721,  588.90914085]),([-144.03976857,    1.64178978,  589.81822066]),([-125.58219355,    2.35938806,  591.63636364]),([-109.1802715 ,    2.52993571,  589.72731712]),([-90.13519079,   2.00058886, 590.72728382]),([-74.31151303,   1.82249473, 590.90910201]),([-56.82747338,   2.54177207, 592.4545621 ]),([-39.35752522,   2.72207561, 592.9091353 ]),([-21.52652758,   1.65290803, 594.00004994]),([ -3.97754121,   2.90387138, 593.54551003]),([ 13.76932976,   2.9114322 , 595.09091464]),([ 31.08580952,   2.90707364, 594.20004272]),([ 47.5087574 ,   2.91147664, 595.1       ]),([ 66.03960904,   2.89897893, 592.54547674]),([ 82.5540826 ,   2.90190169, 593.14290074]),([100.42879347,   2.90698474, 594.18186812]),([117.98353271,   2.1151423 , 593.30004272]),([136.15896884,   2.91276646, 595.36363636]),([153.87762035,   2.73259195, 595.18181818]),([ -1.11567915, -97.47390886, 593.81824285]),([ -1.8162696 , -87.94507523, 594.60002441]),([ -1.16711667, -78.32098062, 593.57147217]),([ -1.6564969 , -68.09627117, 595.09091464]),([ -1.21998062, -57.96098404, 592.30001831]),([ -1.22309111, -48.29136963, 594.10004883]),([ -7.18071638, -44.88624496, 598.80004883]),([ -1.83152625, -28.58628082, 593.8182373 ]),([ -1.65341891, -18.04013408, 594.18186257]),([-2.38718465e-01, -7.13605776e+00,  5.94100049e+02]),([ -0.91399111,   2.24333368, 592.22223579]),([ -1.26924747,  12.23240483, 592.75003815]),([ -1.47376619,  22.41531702, 594.00004439]),([ -1.02577653,  32.21454868, 593.50005493]),([ -0.57377674,  37.97804018, 538.54547674]),([ -0.63227563,  52.14416008, 594.20004272]),([ -0.63014571,  62.17535782, 592.20001221]),([ -1.82844187,  72.02929826, 592.81822621]),([ -1.65495151,  82.62942089, 594.54548784]),([ -1.47500378,  92.64784241, 594.45456765])]

# points = [[-154.3015566739169, 14.29776685888117, 577.0000221946023], [-130.92357982288706, 12.361877528103916, 575.9091242009944], [-109.84913912686434, 10.479419274763627, 577.3636752041904], [-88.07448855313388, 8.58057637648149, 578.3636585582386], [-65.1456229469993, 6.654156121340665, 577.5454711914062], [-42.10764971646395, 4.733159585432573, 576.727294921875], [-20.56534905867143, 2.8233687660910864, 577.0909201882102], [2.391836415637623, 0.7357283722270619, 576.4545676491477], [24.74294489080256, -0.9985503012483771, 575.0000332919034], [47.62744001908736, -2.9048542976379395, 575.0909645774148], [68.90314622358842, -4.795790108767423, 573.2727328213779], [91.79733276367188, -6.516882679679177, 572.7272838245739], [114.49059503728694, -8.555135293440385, 570.5455044833096], [136.61184137517756, -10.6408109664917, 571.727294921875], [160.84658674760297, -13.074495142156428, 572.7272727272727], [182.2080438787287, -15.727754766290838, 568.8182262073864], [207.0548553466797, -17.03106689453125, 569.4000244140625], [228.14845428466796, -19.463539123535156, 568.8000122070313], [253.25570678710938, -20.149084091186523, 567.8182151100852], [269.2584228515625, -23.578112602233887, 568.0000305175781], [55.42946728793058, -197.18501420454547, 578.7273115678267], [57.497575933283024, -177.21400451660156, 576.727294921875], [59.22966038097035, -158.37233248623934, 578.3636641068892], [61.13728401877663, -140.92276000976562, 578.272743918679], [62.142324621027164, -121.33475078235973, 576.2727661132812], [64.12313946810636, -102.86858992143111, 576.909102006392], [65.43955369429155, -84.2959941517223, 574.7273004705256], [66.66296525435014, -65.77646914395419, 574.8182206587358], [68.3945576060902, -47.0714673128995, 574.8182040127841], [69.86699329723011, -29.539445530284535, 574.0909312855114], [72.06902243874289, -10.529438799077814, 575.0909479314631], [73.28159262917258, 7.318910945545543, 575.0909423828125], [74.40809908780184, 26.461584958163176, 573.0909090909091], [76.1558671431108, 44.7063532742587, 574.5454933860085], [77.85080649636008, 63.57825678045099, 573.0000055486506], [79.88976287841797, 80.81198328191584, 574.0000110973011], [81.65036565607244, 100.18132227117366, 573.0], [83.01915879683061, 118.73636973987927, 571.7273004705256], [84.32889834317294, 137.80821228027344, 570.0909590287642], [86.16540804776278, 155.7451865456321, 569.7273060191761]]
# data_show_3d(points)