import os
import matplotlib.pylab as pl
import numpy as np

import ot

import os
import numpy as np
import matplotlib.pyplot as plt
import ot  # Optimal Transport library

def getEMD(xs, xt, num_samples, path,filename):
    n = num_samples  # Number of samples

    # Uniform distribution on samples
    a, b = np.ones((n,)), np.ones((n,))

    # 美化的源和目标分布图
    plt.figure(figsize=(8, 6))
    plt.scatter(xs[:, 0], xs[:, 1], color='blue', marker='+', label='Source samples', s=40, alpha=0.7)
    plt.scatter(xt[:, 0], xt[:, 1], color='red', marker='x', label='Target samples', s=40, alpha=0.7)
    plt.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True)
    plt.title('Source and Target Distributions', fontsize=14, fontweight='bold')
    plt.xlabel('X-axis', fontsize=12)
    plt.ylabel('Y-axis', fontsize=12)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # 保存文件
    savename = os.path.splitext(os.path.basename(path))[0]
    # save_dir = 'PICC'
    save_dir = f'/home/fan/Code/VAE_Synthetic_Steganography/results/{filename}/PCA'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{savename}_distribution.jpg'))
    plt.close()

    # 计算 SWD
    n_seed = 50
    n_projections_arr = np.logspace(0, 3, 25, dtype=int)
    res = np.empty((n_seed, 25))
    for seed in range(n_seed):
        for i, n_projections in enumerate(n_projections_arr):
            res[seed, i] = ot.sliced_wasserstein_distance(xs, xt, a, b, n_projections, seed=seed)

    res_mean = np.mean(res, axis=0)
    res_std = np.std(res, axis=0)

    # 绘制 SWD 曲线（保持不变）
    plt.figure(2)
    plt.plot(n_projections_arr, res_mean, label="SWD")
    plt.fill_between(n_projections_arr, res_mean - 2 * res_std, res_mean + 2 * res_std, alpha=0.5)
    plt.legend()
    plt.xscale('log')
    plt.xlabel("Number of projections")
    plt.ylabel("Distance")
    plt.title('Sliced Wasserstein Distance with 95% confidence interval')

    return res_mean[-1]


if __name__ == '__main__':
    n = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    mu_t = np.array([4, 4])
    cov_t = np.array([[1, -.8], [-.8, 1]])
    xs = ot.datasets.make_2D_samples_gauss(n, mu_s, cov_s)
    xt = ot.datasets.make_2D_samples_gauss(n, mu_t, cov_t)

    getEMD(xs,xt,num_samples=n,path='test.jpg')
