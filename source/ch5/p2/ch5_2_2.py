import numpy as np
import matplotlib.pyplot as plt

def show_data(ax, x0, x1, t):
    for i in range(len(x0)):
        ax.plot([x0[i], x0[i]], [x1[i], x1[i]], [120, t[i]], color='grey')
        ax.plot(x0, x1, t, 'o', color='cornflowerblue', markeredgecolor='black',
                markersize=6, markeredgewidth=0.5)
        ax.view_init(elev=35, azim=75)

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
    
    print(np.round(X0, 2))
    print(np.round(X1, 2))
    print(np.round(T, 2))

    plt.figure(figsize=(6, 5))
    ax = plt.subplot(1, 1, 1, projection='3d')
    show_data(ax, X0, X1, T)
    plt.show()
