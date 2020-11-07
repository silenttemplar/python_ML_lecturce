import numpy as np
import matplotlib.pyplot as plt
from source.ch5.p2.ch5_2_2 import show_data

# 면 표현
def show_plane(ax, w, X0, X1):
    ax0 = np.linspace(X0[0], X0[1], 5)
    ax1 = np.linspace(X1[0], X1[1], 5)
    px0, px1 = np.meshgrid(ax0, ax1)
    y = w[0]*px0 + w[1]*px1 + w[2]
    
    ax.plot_surface(px0, px1, y, rstride=1, cstride=1, alpha=0.3, 
                    color='blue', edgecolor='black')

# 면 MSE
def mse_place(x0, x1, t, w):
    y = w[0]*x0 + w[1]*x1 + w[2]
    mse = np.mean((y-t)**2)
    return mse

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch5_2_data.npz')

    # 나이
    X0 = data['X0']
    X0_min = data['X0_min']
    X0_max = data['X0_max']
    # 몸무게
    X1 = data['X1']
    X1_min = data['X1_min']
    X1_max = data['X1_max']
    # 키
    T = data['T']

    W = [1.5, 1, 90]
    mse = mse_place(X0, X1, T, W)
    print("SD={0:.2f}cm".format(np.sqrt(mse)))

    plt.figure(figsize=(6,5))
    ax = plt.subplot(1,1,1, projection='3d')
    show_plane(ax, W, (X0_min, X0_max), (X1_min, X1_max))
    show_data(ax, X0, X1, T)
    plt.show()
