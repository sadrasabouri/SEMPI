import torch
import torch.nn as nn

from transformers import RobertaModel, RobertaTokenizer

OPENFACE_COLUMN_NAMES = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y', 'eye_lmk_x_0', 'eye_lmk_x_1', 'eye_lmk_x_2', 'eye_lmk_x_3', 'eye_lmk_x_4', 'eye_lmk_x_5', 'eye_lmk_x_6', 'eye_lmk_x_7', 'eye_lmk_x_8', 'eye_lmk_x_9', 'eye_lmk_x_10', 'eye_lmk_x_11', 'eye_lmk_x_12', 'eye_lmk_x_13', 'eye_lmk_x_14', 'eye_lmk_x_15', 'eye_lmk_x_16', 'eye_lmk_x_17', 'eye_lmk_x_18', 'eye_lmk_x_19', 'eye_lmk_x_20', 'eye_lmk_x_21', 'eye_lmk_x_22', 'eye_lmk_x_23', 'eye_lmk_x_24', 'eye_lmk_x_25', 'eye_lmk_x_26', 'eye_lmk_x_27', 'eye_lmk_x_28', 'eye_lmk_x_29', 'eye_lmk_x_30', 'eye_lmk_x_31', 'eye_lmk_x_32', 'eye_lmk_x_33', 'eye_lmk_x_34', 'eye_lmk_x_35', 'eye_lmk_x_36', 'eye_lmk_x_37', 'eye_lmk_x_38', 'eye_lmk_x_39', 'eye_lmk_x_40', 'eye_lmk_x_41', 'eye_lmk_x_42', 'eye_lmk_x_43', 'eye_lmk_x_44', 'eye_lmk_x_45', 'eye_lmk_x_46', 'eye_lmk_x_47', 'eye_lmk_x_48', 'eye_lmk_x_49', 'eye_lmk_x_50', 'eye_lmk_x_51', 'eye_lmk_x_52', 'eye_lmk_x_53', 'eye_lmk_x_54', 'eye_lmk_x_55', 'eye_lmk_y_0', 'eye_lmk_y_1', 'eye_lmk_y_2', 'eye_lmk_y_3', 'eye_lmk_y_4', 'eye_lmk_y_5', 'eye_lmk_y_6', 'eye_lmk_y_7', 'eye_lmk_y_8', 'eye_lmk_y_9', 'eye_lmk_y_10', 'eye_lmk_y_11', 'eye_lmk_y_12', 'eye_lmk_y_13', 'eye_lmk_y_14', 'eye_lmk_y_15', 'eye_lmk_y_16', 'eye_lmk_y_17', 'eye_lmk_y_18', 'eye_lmk_y_19', 'eye_lmk_y_20', 'eye_lmk_y_21', 'eye_lmk_y_22', 'eye_lmk_y_23', 'eye_lmk_y_24', 'eye_lmk_y_25', 'eye_lmk_y_26', 'eye_lmk_y_27', 'eye_lmk_y_28', 'eye_lmk_y_29', 'eye_lmk_y_30', 'eye_lmk_y_31', 'eye_lmk_y_32', 'eye_lmk_y_33', 'eye_lmk_y_34', 'eye_lmk_y_35', 'eye_lmk_y_36', 'eye_lmk_y_37', 'eye_lmk_y_38', 'eye_lmk_y_39', 'eye_lmk_y_40', 'eye_lmk_y_41', 'eye_lmk_y_42', 'eye_lmk_y_43', 'eye_lmk_y_44', 'eye_lmk_y_45', 'eye_lmk_y_46', 'eye_lmk_y_47', 'eye_lmk_y_48', 'eye_lmk_y_49', 'eye_lmk_y_50', 'eye_lmk_y_51', 'eye_lmk_y_52', 'eye_lmk_y_53', 'eye_lmk_y_54', 'eye_lmk_y_55', 'eye_lmk_X_0', 'eye_lmk_X_1', 'eye_lmk_X_2', 'eye_lmk_X_3', 'eye_lmk_X_4', 'eye_lmk_X_5', 'eye_lmk_X_6', 'eye_lmk_X_7', 'eye_lmk_X_8', 'eye_lmk_X_9', 'eye_lmk_X_10', 'eye_lmk_X_11', 'eye_lmk_X_12', 'eye_lmk_X_13', 'eye_lmk_X_14', 'eye_lmk_X_15', 'eye_lmk_X_16', 'eye_lmk_X_17', 'eye_lmk_X_18', 'eye_lmk_X_19', 'eye_lmk_X_20', 'eye_lmk_X_21', 'eye_lmk_X_22', 'eye_lmk_X_23', 'eye_lmk_X_24', 'eye_lmk_X_25', 'eye_lmk_X_26', 'eye_lmk_X_27', 'eye_lmk_X_28', 'eye_lmk_X_29', 'eye_lmk_X_30', 'eye_lmk_X_31', 'eye_lmk_X_32', 'eye_lmk_X_33', 'eye_lmk_X_34', 'eye_lmk_X_35', 'eye_lmk_X_36', 'eye_lmk_X_37', 'eye_lmk_X_38', 'eye_lmk_X_39', 'eye_lmk_X_40', 'eye_lmk_X_41', 'eye_lmk_X_42', 'eye_lmk_X_43', 'eye_lmk_X_44', 'eye_lmk_X_45', 'eye_lmk_X_46', 'eye_lmk_X_47', 'eye_lmk_X_48', 'eye_lmk_X_49', 'eye_lmk_X_50', 'eye_lmk_X_51', 'eye_lmk_X_52', 'eye_lmk_X_53', 'eye_lmk_X_54', 'eye_lmk_X_55', 'eye_lmk_Y_0', 'eye_lmk_Y_1', 'eye_lmk_Y_2', 'eye_lmk_Y_3', 'eye_lmk_Y_4', 'eye_lmk_Y_5', 'eye_lmk_Y_6', 'eye_lmk_Y_7', 'eye_lmk_Y_8', 'eye_lmk_Y_9', 'eye_lmk_Y_10', 'eye_lmk_Y_11', 'eye_lmk_Y_12', 'eye_lmk_Y_13', 'eye_lmk_Y_14', 'eye_lmk_Y_15', 'eye_lmk_Y_16', 'eye_lmk_Y_17', 'eye_lmk_Y_18', 'eye_lmk_Y_19', 'eye_lmk_Y_20', 'eye_lmk_Y_21', 'eye_lmk_Y_22', 'eye_lmk_Y_23', 'eye_lmk_Y_24', 'eye_lmk_Y_25', 'eye_lmk_Y_26', 'eye_lmk_Y_27', 'eye_lmk_Y_28', 'eye_lmk_Y_29', 'eye_lmk_Y_30', 'eye_lmk_Y_31', 'eye_lmk_Y_32', 'eye_lmk_Y_33', 'eye_lmk_Y_34', 'eye_lmk_Y_35', 'eye_lmk_Y_36', 'eye_lmk_Y_37', 'eye_lmk_Y_38', 'eye_lmk_Y_39', 'eye_lmk_Y_40', 'eye_lmk_Y_41', 'eye_lmk_Y_42', 'eye_lmk_Y_43', 'eye_lmk_Y_44', 'eye_lmk_Y_45', 'eye_lmk_Y_46', 'eye_lmk_Y_47', 'eye_lmk_Y_48', 'eye_lmk_Y_49', 'eye_lmk_Y_50', 'eye_lmk_Y_51', 'eye_lmk_Y_52', 'eye_lmk_Y_53', 'eye_lmk_Y_54', 'eye_lmk_Y_55', 'eye_lmk_Z_0', 'eye_lmk_Z_1', 'eye_lmk_Z_2', 'eye_lmk_Z_3', 'eye_lmk_Z_4', 'eye_lmk_Z_5', 'eye_lmk_Z_6', 'eye_lmk_Z_7', 'eye_lmk_Z_8', 'eye_lmk_Z_9', 'eye_lmk_Z_10', 'eye_lmk_Z_11', 'eye_lmk_Z_12', 'eye_lmk_Z_13', 'eye_lmk_Z_14', 'eye_lmk_Z_15', 'eye_lmk_Z_16', 'eye_lmk_Z_17', 'eye_lmk_Z_18', 'eye_lmk_Z_19', 'eye_lmk_Z_20', 'eye_lmk_Z_21', 'eye_lmk_Z_22', 'eye_lmk_Z_23', 'eye_lmk_Z_24', 'eye_lmk_Z_25', 'eye_lmk_Z_26', 'eye_lmk_Z_27', 'eye_lmk_Z_28', 'eye_lmk_Z_29', 'eye_lmk_Z_30', 'eye_lmk_Z_31', 'eye_lmk_Z_32', 'eye_lmk_Z_33', 'eye_lmk_Z_34', 'eye_lmk_Z_35', 'eye_lmk_Z_36', 'eye_lmk_Z_37', 'eye_lmk_Z_38', 'eye_lmk_Z_39', 'eye_lmk_Z_40', 'eye_lmk_Z_41', 'eye_lmk_Z_42', 'eye_lmk_Z_43', 'eye_lmk_Z_44', 'eye_lmk_Z_45', 'eye_lmk_Z_46', 'eye_lmk_Z_47', 'eye_lmk_Z_48', 'eye_lmk_Z_49', 'eye_lmk_Z_50', 'eye_lmk_Z_51', 'eye_lmk_Z_52', 'eye_lmk_Z_53', 'eye_lmk_Z_54', 'eye_lmk_Z_55', 'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']

class Classifier(nn.Module):
    def __init__(self, in_size, hidden_size, dropout, num_classes, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(dropout)
        #import pdb; pdb.set_trace()
        hidden_size = int(hidden_size)
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        if self.config.activation_fn == "relu":
            self.activation_fn = nn.ReLU()
        elif (self.config.activation_fn == "leaky_relu"):
            self.activation_fn = nn.LeakyReLU()
        elif self.config.activation_fn == "tanh":
            self.activation_fn = nn.Tanh()
    def forward(self, x):
        x = self.dropout(self.fc1(x))
        #x = torch.tanh(x)
        x = self.activation_fn(x)
        if not (self.config.expnum == 1 or self.config.expnum == 4 or self.config.expnum == 3 or self.config.expnum == 10):
            x = self.dropout(self.fc2(x))
        #x = torch.tanh(x)
        x = self.activation_fn(x)
        x = self.fc3(x)
        if self.config.extra_dropout:
            x = self.dropout(x)
        return x
import numpy as np
MAX_VEC = np.array([ 3.2483e-01,  2.0257e-01, -9.2535e-01,  1.7258e-01,  2.0555e-01,
        -9.6917e-01,  2.5606e-01,  2.0719e-01,  9.3451e+01,  9.5400e+01,
         1.0009e+02,  1.0475e+02,  1.0668e+02,  1.0498e+02,  1.0005e+02,
         9.5377e+01,  8.6615e+01,  8.9760e+01,  9.4472e+01,  9.9932e+01,
         1.0492e+02,  1.0827e+02,  1.1025e+02,  1.0748e+02,  1.0357e+02,
         9.8928e+01,  9.4170e+01,  8.9734e+01,  9.7726e+01,  9.9840e+01,
         1.0196e+02,  1.0285e+02,  1.0197e+02,  9.9860e+01,  9.7747e+01,
         9.6851e+01,  1.5204e+02,  1.5411e+02,  1.5909e+02,  1.6431e+02,
         1.6657e+02,  1.6454e+02,  1.5943e+02,  1.5382e+02,  1.4697e+02,
         1.4997e+02,  1.5431e+02,  1.5975e+02,  1.6514e+02,  1.6948e+02,
         1.7259e+02,  1.6957e+02,  1.6532e+02,  1.6036e+02,  1.5528e+02,
         1.5037e+02,  1.5700e+02,  1.5921e+02,  1.6139e+02,  1.6225e+02,
         1.6128e+02,  1.5907e+02,  1.5690e+02,  1.5604e+02,  1.2151e+02,
         1.1741e+02,  1.1572e+02,  1.1743e+02,  1.2154e+02,  1.2642e+02,
         1.2817e+02,  1.2614e+02,  1.2379e+02,  1.2197e+02,  1.2064e+02,
         1.2003e+02,  1.2061e+02,  1.2199e+02,  1.2382e+02,  1.2571e+02,
         1.2667e+02,  1.2709e+02,  1.2655e+02,  1.2520e+02,  1.2370e+02,
         1.2444e+02,  1.2370e+02,  1.2194e+02,  1.2018e+02,  1.1944e+02,
         1.2017e+02,  1.2193e+02,  1.2121e+02,  1.1705e+02,  1.1556e+02,
         1.1730e+02,  1.2162e+02,  1.2667e+02,  1.2864e+02,  1.2675e+02,
         1.2391e+02,  1.2123e+02,  1.2000e+02,  1.1949e+02,  1.2014e+02,
         1.2136e+02,  1.2318e+02,  1.2535e+02,  1.2650e+02,  1.2686e+02,
         1.2639e+02,  1.2544e+02,  1.2360e+02,  1.2458e+02,  1.2373e+02,
         1.2157e+02,  1.1974e+02,  1.1902e+02,  1.1965e+02,  1.2139e+02,
        -2.9723e+01, -2.8079e+01, -2.4189e+01, -2.0328e+01, -1.8730e+01,
        -2.0211e+01, -2.4330e+01, -2.8181e+01, -3.6153e+01, -3.2902e+01,
        -2.8711e+01, -2.4111e+01, -2.0047e+01, -1.7379e+01, -1.5800e+01,
        -1.8064e+01, -2.1217e+01, -2.5045e+01, -2.9098e+01, -3.3032e+01,
        -2.6306e+01, -2.4557e+01, -2.2783e+01, -2.2032e+01, -2.2734e+01,
        -2.4502e+01, -2.6257e+01, -2.7000e+01,  2.4817e+01,  2.6657e+01,
         3.1028e+01,  3.5366e+01,  3.7089e+01,  3.5194e+01,  3.0838e+01,
         2.6245e+01,  1.9772e+01,  2.2423e+01,  2.6040e+01,  3.0604e+01,
         3.5323e+01,  3.9457e+01,  4.2774e+01,  3.9523e+01,  3.5430e+01,
         3.0983e+01,  2.6481e+01,  2.2523e+01,  2.9072e+01,  3.0902e+01,
         3.2740e+01,  3.3540e+01,  3.2809e+01,  3.0985e+01,  2.9126e+01,
         2.8347e+01, -5.6915e+00, -9.8787e+00, -1.1696e+01, -1.0030e+01,
        -5.8489e+00, -1.3340e+00,  1.6596e-01, -1.5319e+00, -4.1128e+00,
        -5.8511e+00, -7.1000e+00, -7.6957e+00, -7.1596e+00, -5.8745e+00,
        -3.6702e+00, -2.0128e+00, -1.1553e+00, -8.0851e-01, -1.2660e+00,
        -2.4170e+00, -3.8787e+00, -3.1170e+00, -3.9234e+00, -5.8383e+00,
        -7.6830e+00, -8.4021e+00, -7.6745e+00, -5.7766e+00, -5.7556e+00,
        -1.0031e+01, -1.1762e+01, -9.9044e+00, -5.5000e+00, -1.1889e+00,
         4.7778e-01, -1.1178e+00, -3.9809e+00, -6.2911e+00, -7.8044e+00,
        -8.3556e+00, -7.6800e+00, -6.3933e+00, -4.2822e+00, -2.3400e+00,
        -1.3400e+00, -1.0133e+00, -1.4267e+00, -2.2356e+00, -3.7889e+00,
        -2.9622e+00, -3.6911e+00, -5.5489e+00, -7.4356e+00, -8.2467e+00,
        -7.5111e+00, -5.6600e+00,  2.2766e+02,  2.2983e+02,  2.3130e+02,
         2.3120e+02,  2.2958e+02,  2.2757e+02,  2.2593e+02,  2.2603e+02,
         2.3047e+02,  2.2862e+02,  2.2730e+02,  2.2704e+02,  2.2804e+02,
         2.2972e+02,  2.3147e+02,  2.2878e+02,  2.2641e+02,  2.2532e+02,
         2.2586e+02,  2.2773e+02,  2.2824e+02,  2.2821e+02,  2.2884e+02,
         2.2977e+02,  2.3046e+02,  2.3051e+02,  2.2988e+02,  2.2895e+02,
         2.4827e+02,  2.4970e+02,  2.5068e+02,  2.5063e+02,  2.4957e+02,
         2.4813e+02,  2.4716e+02,  2.4736e+02,  2.5011e+02,  2.4869e+02,
         2.4749e+02,  2.4725e+02,  2.4846e+02,  2.5069e+02,  2.5336e+02,
         2.5013e+02,  2.4758e+02,  2.4624e+02,  2.4646e+02,  2.4807e+02,
         2.4898e+02,  2.4896e+02,  2.4937e+02,  2.4999e+02,  2.5042e+02,
         2.5046e+02,  2.5004e+02,  2.4943e+02, -2.6383e-01,  3.2140e+01,
         2.4434e+02,  9.5744e-02,  1.3541e-01,  4.2766e-02,  1.8106e-01,
         1.2689e-01,  1.3468e+00,  7.9362e-02,  4.6085e-01,  6.7077e-01,
         1.6213e-01,  1.0187e+00,  7.5746e-01,  2.0694e+00,  1.9957e-01,
         6.8872e-01,  1.0596e-01,  1.6830e-01,  4.6809e-01,  3.7957e-01,
         4.9085e-01,  1.3559e-01,  1.4894e-01,  1.0000e+00,  1.0000e+00,
         7.4468e-01,  1.0000e+00,  2.1277e-01,  1.0000e+00,  9.6610e-01,
         1.0000e+00,  2.2222e-01,  4.2553e-01,  1.4894e-01,  1.0000e+00,
         1.4894e-01,  2.4444e-01,  0.0000e+00,  2.5532e-01])
MIN_VEC = np.array([ 7.0334e-02, -3.5600e-02, -9.9080e-01, -1.3505e-01, -7.9919e-02,
        -9.9743e-01, -3.3308e-02, -5.9319e-02,  8.7630e+01,  8.9168e+01,
         9.3447e+01,  9.7981e+01,  1.0010e+02,  9.8855e+01,  9.4272e+01,
         8.9745e+01,  8.0957e+01,  8.4338e+01,  8.8517e+01,  9.3183e+01,
         9.7730e+01,  1.0136e+02,  1.0395e+02,  1.0143e+02,  9.7562e+01,
         9.2998e+01,  8.8455e+01,  8.4245e+01,  9.1974e+01,  9.3923e+01,
         9.5764e+01,  9.6426e+01,  9.5513e+01,  9.3564e+01,  9.1711e+01,
         9.1062e+01,  1.5001e+02,  1.5190e+02,  1.5580e+02,  1.5959e+02,
         1.6146e+02,  1.6029e+02,  1.5664e+02,  1.5155e+02,  1.4437e+02,
         1.4767e+02,  1.5196e+02,  1.5675e+02,  1.6061e+02,  1.6319e+02,
         1.6461e+02,  1.6327e+02,  1.6080e+02,  1.5741e+02,  1.5279e+02,
         1.4794e+02,  1.5430e+02,  1.5615e+02,  1.5769e+02,  1.5820e+02,
         1.5739e+02,  1.5573e+02,  1.5420e+02,  1.5346e+02,  1.1793e+02,
         1.1323e+02,  1.1126e+02,  1.1318e+02,  1.1788e+02,  1.2293e+02,
         1.2455e+02,  1.2263e+02,  1.1964e+02,  1.1745e+02,  1.1612e+02,
         1.1543e+02,  1.1616e+02,  1.1792e+02,  1.1997e+02,  1.2177e+02,
         1.2284e+02,  1.2333e+02,  1.2291e+02,  1.2175e+02,  1.1998e+02,
         1.2079e+02,  1.1994e+02,  1.1795e+02,  1.1595e+02,  1.1513e+02,
         1.1597e+02,  1.1798e+02,  1.1786e+02,  1.1298e+02,  1.1111e+02,
         1.1335e+02,  1.1838e+02,  1.2326e+02,  1.2513e+02,  1.2327e+02,
         1.1938e+02,  1.1678e+02,  1.1508e+02,  1.1456e+02,  1.1569e+02,
         1.1700e+02,  1.1901e+02,  1.2164e+02,  1.2280e+02,  1.2284e+02,
         1.2226e+02,  1.2139e+02,  1.2043e+02,  1.2143e+02,  1.2059e+02,
         1.1840e+02,  1.1612e+02,  1.1513e+02,  1.1597e+02,  1.1815e+02,
        -3.9355e+01, -3.8215e+01, -3.4230e+01, -2.9740e+01, -2.7455e+01,
        -2.8428e+01, -3.2638e+01, -3.7021e+01, -4.6381e+01, -4.2715e+01,
        -3.8423e+01, -3.3847e+01, -2.9566e+01, -2.6238e+01, -2.3868e+01,
        -2.6055e+01, -2.9534e+01, -3.3760e+01, -3.8236e+01, -4.2651e+01,
        -3.5219e+01, -3.3313e+01, -3.1609e+01, -3.1091e+01, -3.2079e+01,
        -3.4011e+01, -3.5717e+01, -3.6215e+01,  1.8564e+01,  2.0227e+01,
         2.4307e+01,  2.8453e+01,  3.0196e+01,  2.8482e+01,  2.4353e+01,
         1.9924e+01,  1.3920e+01,  1.6602e+01,  2.0213e+01,  2.4700e+01,
         2.8973e+01,  3.2391e+01,  3.4804e+01,  3.2371e+01,  2.8982e+01,
         2.5062e+01,  2.0856e+01,  1.6842e+01,  2.2407e+01,  2.4178e+01,
         2.5953e+01,  2.6693e+01,  2.5940e+01,  2.4173e+01,  2.2409e+01,
         2.1673e+01, -8.9692e+00, -1.3185e+01, -1.4941e+01, -1.3226e+01,
        -9.0385e+00, -4.5256e+00, -3.0590e+00, -4.7846e+00, -7.6077e+00,
        -9.4590e+00, -1.0538e+01, -1.1100e+01, -1.0500e+01, -9.0886e+00,
        -7.1727e+00, -5.4532e+00, -4.4809e+00, -4.1615e+00, -4.5615e+00,
        -5.6410e+00, -7.1846e+00, -6.4513e+00, -7.2077e+00, -9.0103e+00,
        -1.0782e+01, -1.1518e+01, -1.0767e+01, -8.9692e+00, -8.7159e+00,
        -1.2957e+01, -1.4652e+01, -1.2727e+01, -8.3500e+00, -4.0886e+00,
        -2.4636e+00, -4.0614e+00, -7.4750e+00, -9.6591e+00, -1.1084e+01,
        -1.1516e+01, -1.0634e+01, -9.4831e+00, -7.8864e+00, -5.5102e+00,
        -4.4932e+00, -4.5383e+00, -4.9114e+00, -5.6818e+00, -6.5523e+00,
        -5.6795e+00, -6.4250e+00, -8.3500e+00, -1.0323e+01, -1.1184e+01,
        -1.0434e+01, -8.5182e+00,  1.9217e+02,  1.9178e+02,  1.9179e+02,
         1.9218e+02,  1.9273e+02,  1.9343e+02,  1.9312e+02,  1.9272e+02,
         1.9543e+02,  1.9284e+02,  1.9072e+02,  1.8999e+02,  1.9112e+02,
         1.9313e+02,  1.9545e+02,  1.9394e+02,  1.9216e+02,  1.9146e+02,
         1.9205e+02,  1.9364e+02,  1.9336e+02,  1.9352e+02,  1.9352e+02,
         1.9335e+02,  1.9311e+02,  1.9295e+02,  1.9295e+02,  1.9313e+02,
         1.9326e+02,  1.9319e+02,  1.9357e+02,  1.9419e+02,  1.9467e+02,
         1.9472e+02,  1.9434e+02,  1.9399e+02,  1.9516e+02,  1.9332e+02,
         1.9193e+02,  1.9165e+02,  1.9317e+02,  1.9586e+02,  1.9896e+02,
         1.9641e+02,  1.9401e+02,  1.9256e+02,  1.9254e+02,  1.9383e+02,
         1.9469e+02,  1.9494e+02,  1.9510e+02,  1.9508e+02,  1.9489e+02,
         1.9462e+02,  1.9446e+02,  1.9449e+02, -2.7936e+00,  2.6409e+01,
         2.2061e+02, -3.0566e-01, -3.0404e-01, -2.3897e-02,  9.5227e-02,
         3.1364e-02,  0.0000e+00,  1.5682e-02,  0.0000e+00,  0.0000e+00,
         1.3182e-02,  0.0000e+00,  0.0000e+00,  3.0682e-01,  4.8205e-02,
         1.3745e-01,  3.4468e-02,  1.7045e-02,  1.0773e-01,  6.6591e-02,
         6.7458e-02,  0.0000e+00,  0.0000e+00,  0.0000e+00,  6.7797e-02,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00])



class EarlyFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.MAX_VEC = torch.tensor(MAX_VEC, dtype=torch.float32).cuda()
        self.MAX_VEC = torch.maximum(self.MAX_VEC, torch.tensor(1).cuda())
        self.MIN_VEC = torch.tensor(MIN_VEC, dtype=torch.float32).cuda()


        if self.config.openfacefeat == 0:
            self.filteredcolumns = []
        if self.config.openfacefeat == 1:
            #all features
            self.filteredcolumns = OPENFACE_COLUMN_NAMES
        elif self.config.openfacefeat == 2:
            self.filteredcolumns = [i for i in OPENFACE_COLUMN_NAMES if "AU" in i]
        elif self.config.openfacefeat == 3:
            self.filteredcolumns = [i for i in OPENFACE_COLUMN_NAMES if "pose" in i]
        elif self.config.openfacefeat == 4:
            #pose and aus
            self.filteredcolumns = [i for i in OPENFACE_COLUMN_NAMES if "pose" in i] + [i for i in OPENFACE_COLUMN_NAMES if "AU" in i]
        elif self.config.openfacefeat == 5:
            # gaze columns 
            self.filteredcolumns = [i for i in OPENFACE_COLUMN_NAMES if "gaze" in i]
        
        if self.config.openfacefeat_extramlp == 1:
            # Process facial features with MLP to get intermediate representation
            self.extra_mlp = nn.Sequential(
                nn.Linear(len(self.filteredcolumns), self.config.openfacefeat_extramlp_dim),
                (nn.Tanh() if self.config.activation_fn == "tanh" else nn.LeakyReLU()))

            self.out = Classifier(in_size=config.hidden_size, hidden_size=config.hidden_size, 
                              dropout=config.dropout, num_classes=config.num_labels, config = config)


        # Cross-attention module
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=config.dropout
        )

        # Process audio features with MLP to get intermediate representation
        self.audio_mlp = nn.Sequential(
            nn.Linear(768, config.hidden_size),  # 768 is the audio feature size
            (nn.Tanh() if self.config.activation_fn == "tanh" else nn.LeakyReLU()))
        
        # MLP after concatenation of intermediate features
        self.fusion_mlp = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            (nn.Tanh() if self.config.activation_fn == "tanh" else nn.LeakyReLU()))
        

    def forward(self, audio_paths, openfacefeat_):

        B = audio_paths.size(0) # Batch size
        audio_features = audio_paths.mean(dim=2)

        openfacefeat =  [i.mean(0) for i in openfacefeat_]
        openfacefeat = torch.stack(openfacefeat)
        openfacefeat = openfacefeat.cuda()
        openfacefeat = (openfacefeat - self.MIN_VEC) / (self.MAX_VEC - self.MIN_VEC)
        openfacefeat = openfacefeat - 0.5
        openfacefeat_filtered = openfacefeat[:, [OPENFACE_COLUMN_NAMES.index(i) for i in self.filteredcolumns]]

        # Append facial and audio features
        fusion_vec = []
        openfacefeat_filtered = self.extra_mlp(openfacefeat_filtered) # (B, hidden_size)
        audio_intermediate = self.audio_mlp(audio_features) # (B, hidden_size)
        fusion_vec.append(openfacefeat_filtered)
        fusion_vec.append(audio_intermediate)
        
        # Path 1: Concatenation of extracted ficial and audio features followed by MLP
        fusion_vec = torch.cat(fusion_vec, dim=1) # (B, 2 x hidden_size)
        concat_output = self.fusion_mlp(fusion_vec) # (B, hidden_size)

        # Path 2: Cross-attention between facial and audio features
        # Reshape for attention (seq_len, B, hidden_size)
        facial_attention = openfacefeat_filtered.unsqueeze(0)  # (1 x B x hidden_size)
        audio_attention = audio_intermediate.unsqueeze(0)  # (1 x B x hidden_size)

        # Apply cross attention (facial features attending to audio features)
        attention_output, _ = self.cross_attention(
            query=facial_attention,
            key=audio_attention,
            value=audio_attention
        )
        attention_output = attention_output.squeeze(0)  # (B x hidden_size)

        # Combine outputs from both paths
        combined_output = concat_output + attention_output # (B x hidden_size)

        return self.out(combined_output) # (B x num_classes)
        #############
    

class EarlyFusion2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.out = Classifier(in_size=config.hidden_size, hidden_size=config.hidden_size, 
                          dropout=config.dropout, num_classes=config.num_labels, config=config)

        # Cross-attention module
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=config.dropout
        )

        # Process video features with MLP
        self.video_mlp = nn.Sequential(
            nn.Linear(1024, config.hidden_size),  # 1024 is the video feature size
            (nn.Tanh() if self.config.activation_fn == "tanh" else nn.LeakyReLU()))

        # Process audio features with MLP
        self.audio_mlp = nn.Sequential(
            nn.Linear(768, config.hidden_size),  # 768 is the audio feature size
            (nn.Tanh() if self.config.activation_fn == "tanh" else nn.LeakyReLU()))
        
        # MLP after concatenation
        self.fusion_mlp = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            (nn.Tanh() if self.config.activation_fn == "tanh" else nn.LeakyReLU()))
        
    def forward(self, audio_paths, video_paths):
        B = audio_paths.size(0)  # Batch size
        audio_features = audio_paths.mean(dim=2)

        if video_paths.dim() > 2:
            video_features = video_paths.mean(dim=2)

        # Process video features
        video_intermediate = self.video_mlp(video_features)  # (B, hidden_size)
        
        # Process audio features
        audio_intermediate = self.audio_mlp(audio_features)  # (B, hidden_size)
        
        # Append video and audio features
        fusion_vec = []
        fusion_vec.append(video_intermediate)
        fusion_vec.append(audio_intermediate)
        
        # Path 1: Concatenation followed by MLP
        fusion_vec = torch.cat(fusion_vec, dim=1)  # (B, 2 x hidden_size)
        concat_output = self.fusion_mlp(fusion_vec)  # (B, hidden_size)

        # Path 2: Cross-attention between video and audio features
        # Reshape for attention (seq_len, B, hidden_size)
        video_attention = video_intermediate.unsqueeze(0)  # (1 x B x hidden_size)
        audio_attention = audio_intermediate.unsqueeze(0)  # (1 x B x hidden_size)

        # Apply cross attention (video features attending to audio features)
        attention_output, _ = self.cross_attention(
            query=video_attention,
            key=audio_attention,
            value=audio_attention
        )
        attention_output = attention_output.squeeze(0)  # (B x hidden_size)

        # Combine outputs from both paths
        combined_output = concat_output + attention_output  # (B x hidden_size)

        return self.out(combined_output)  # (B x num_classes)
    

class EarlyFusion3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.MAX_VEC = torch.tensor(MAX_VEC, dtype=torch.float32).cuda()
        self.MAX_VEC = torch.maximum(self.MAX_VEC, torch.tensor(1).cuda())
        self.MIN_VEC = torch.tensor(MIN_VEC, dtype=torch.float32).cuda()


        if self.config.openfacefeat == 0:
            self.filteredcolumns = []
        if self.config.openfacefeat == 1:
            #all features
            self.filteredcolumns = OPENFACE_COLUMN_NAMES
        elif self.config.openfacefeat == 2:
            self.filteredcolumns = [i for i in OPENFACE_COLUMN_NAMES if "AU" in i]
        elif self.config.openfacefeat == 3:
            self.filteredcolumns = [i for i in OPENFACE_COLUMN_NAMES if "pose" in i]
        elif self.config.openfacefeat == 4:
            #pose and aus
            self.filteredcolumns = [i for i in OPENFACE_COLUMN_NAMES if "pose" in i] + [i for i in OPENFACE_COLUMN_NAMES if "AU" in i]
        elif self.config.openfacefeat == 5:
            # gaze columns 
            self.filteredcolumns = [i for i in OPENFACE_COLUMN_NAMES if "gaze" in i]
        
        if self.config.openfacefeat_extramlp == 1:
            # Process facial features with MLP to get intermediate representation
            self.extra_mlp = nn.Sequential(
                nn.Linear(len(self.filteredcolumns), self.config.openfacefeat_extramlp_dim),
                (nn.Tanh() if self.config.activation_fn == "tanh" else nn.LeakyReLU()))

            self.out = Classifier(in_size=config.hidden_size, hidden_size=config.hidden_size, 
                              dropout=config.dropout, num_classes=config.num_labels, config = config)

        # Cross-attention module
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=config.dropout
        )

        # Process video features with MLP to get intermediate representation
        self.video_mlp = nn.Sequential(
            nn.Linear(1024, config.hidden_size),  # 1024 is the video feature size
            (nn.Tanh() if self.config.activation_fn == "tanh" else nn.LeakyReLU()))

        # Process audio features with MLP to get intermediate representation
        self.audio_mlp = nn.Sequential(
            nn.Linear(768, config.hidden_size),  # 768 is the audio feature size
            (nn.Tanh() if self.config.activation_fn == "tanh" else nn.LeakyReLU()))
        
        # MLP before cross attention
        self.fusion_mlp = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            (nn.Tanh() if self.config.activation_fn == "tanh" else nn.LeakyReLU()))
        
        # MLP after concatenation of intermediate features
        self.fusion_mlp2 = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            (nn.Tanh() if self.config.activation_fn == "tanh" else nn.LeakyReLU()))
        

    def forward(self, audio_paths, openfacefeat_, video_paths):

        B = audio_paths.size(0) # Batch size
        audio_features = audio_paths.mean(dim=2)

        openfacefeat =  [i.mean(0) for i in openfacefeat_]
        openfacefeat = torch.stack(openfacefeat)
        openfacefeat = openfacefeat.cuda()
        openfacefeat = (openfacefeat - self.MIN_VEC) / (self.MAX_VEC - self.MIN_VEC)
        openfacefeat = openfacefeat - 0.5
        openfacefeat_filtered = openfacefeat[:, [OPENFACE_COLUMN_NAMES.index(i) for i in self.filteredcolumns]]

        if video_paths.dim() > 2:
            video_features = video_paths.mean(dim=2)  # Average across time dimension if present

        # Append video, facial, and audio features
        fusion_vec = []
        video_intermediate = self.video_mlp(video_features)  # (B, hidden_size)
        openfacefeat_filtered = self.extra_mlp(openfacefeat_filtered) # (B, hidden_size)
        audio_intermediate = self.audio_mlp(audio_features) # (B, hidden_size)
        fusion_vec.append(video_intermediate)
        fusion_vec.append(openfacefeat_filtered)
        fusion_vec.append(audio_intermediate)
      
        # Path 1: Concatenation followed by MLP
        fusion_vec = torch.cat(fusion_vec, dim=1)  # (B, 3 x hidden_size)
        concat_output = self.fusion_mlp2(fusion_vec)  # (B, hidden_size)

        # Path 2: Cross-attention between video and audio features
        fusion_attention = []
        fusion_attention.append(video_intermediate)
        fusion_attention.append(openfacefeat_filtered)
        fusion_attention = torch.cat(fusion_attention, dim=1) # (B, 2 x hidden_size)
        attention_input = self.fusion_mlp(fusion_attention)  # (B, hidden_size)

        # Reshape for attention (seq_len, B, hidden_size)
        video_openface_attention = attention_input.unsqueeze(0)  # (1 x B x hidden_size)
        audio_attention = audio_intermediate.unsqueeze(0)  # (1 x B x hidden_size)

        # Apply cross attention (video+facial features attending to audio features)
        attention_output, _ = self.cross_attention(
            query=video_openface_attention,
            key=audio_attention,
            value=audio_attention
        )
        attention_output = attention_output.squeeze(0)  # (B x hidden_size)

        # Combine outputs from both paths
        combined_output = concat_output + attention_output # (B x hidden_size)

        return self.out(combined_output) # (B x num_classes)
        #############
