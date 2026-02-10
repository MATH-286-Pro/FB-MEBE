import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import wandb
from typing import Optional
import io
from PIL import Image
from scipy.spatial import KDTree


def PLOT_VALUE(x,y,r,
               s=8.0,
               xlabel='vx (m/s)', 
               ylabel='vy (m/s)',
               colorbar_label='Reward',
               ):
    # 格式转化
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if isinstance(r, torch.Tensor):
        r = r.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    ax.set_facecolor('white')
    scatter = ax.scatter(x, y, c=r, cmap='viridis', s=s, alpha=1.0, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax, label=colorbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')  # Use set_aspect instead of plt.axis('equal')
    ax.set_xlim(-3, 3)      # Set limits after aspect to ensure they take effect
    ax.set_ylim(-3, 3)
    plt.tight_layout()

    # Convert to PIL Image for wandb
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    pil_image = Image.open(buf)
    pil_image.load()  # Load the image so it doesn't disappear when buf is closed
    buf.close()
    
    plt.close(fig)
    return pil_image

def PLOT_DENSITY(x, 
                 y, 
                 radius=None, 
                 k_neighbors=50, 
                 s=8.0,):

    
    # 格式转化
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    x = np.asarray(x).ravel() # flatten to 1D
    y = np.asarray(y).ravel() # flatten to 1D

    # 计算每个点的局部密度
    points = np.column_stack([x, y])
    tree = KDTree(points)
    
    if radius is not None:
        # 使用半径搜索
        density = np.array([len(tree.query_ball_point(point, radius)) for point in points])
    else:
        # 使用k近邻
        density = np.ones(len(points)) * k_neighbors
        # 可选：计算到第k个邻居的平均距离的倒数作为密度
        distances, _ = tree.query(points, k=k_neighbors+1)
        density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-8)
    
    xlabel = "vx (m/s)"
    ylabel = "vy (m/s)"
    colorbar_label = "Density"

    # 使用 PLOT_REWARD 绘制
    fig = PLOT_VALUE(x, y, density, s=s, 
                     xlabel=xlabel, ylabel=ylabel, 
                     colorbar_label=colorbar_label)
    return fig

