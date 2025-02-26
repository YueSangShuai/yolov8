import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import ceil
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os
import time


# fig, ax = plt.subplots(figsize=(48, 6))  #创建图像并设置画布大小 plt.figure(figsize=figsize)
# plt.tick_params(axis='both', which='major', labelsize=14)  # 设置主刻度标签的大小
# plt.tick_params(axis='both', which='minor', labelsize=10)  # 设置次刻度标签的大小
# ax.set_xticks(index)  # 设置刻度位置
# ax.set_xticklabels(index, rotation=45) #刻度旋转
# plt.tight_layout() #用于自动调整子图参数，使图形中的子图、标签、标题等不重叠，并且整体布局更加紧凑美观
# ax.tick_params(axis='x', labelsize=7)
# ax.vlines(index, ymin=0, ymax=data, colors='gray', linestyles='--')
# plt.title('Box Plot per Channel') #设置标题
# plt.xlabel('Channel') #设置x轴标签
# plt.ylabel('Value') #设置y轴标签


def plot_line_fig(data, path):
    # 确保数据可以转换为一维数组
    if isinstance(data, torch.Tensor):
        if data.requires_grad:
            data = data.detach().cpu().numpy()
        else:
            data = data.cpu().numpy()
        
    data = np.array(data).flatten()
    # 绘制折线图
    plt.plot(data, marker='o')  # 使用'o'标记每个数据点
    
    # 设置图表标题和坐标轴标签
    plt.title(path)
    plt.xlabel("Index")
    plt.ylabel("Value")
    
    try:
        plt.tight_layout()  # 自动调整布局
    except Exception as e:
        print(f"Warning: tight_layout failed with error: {e}")
    
    # 保存图表
    plt.savefig(path)
    plt.close()
    print('saving:  ', path)

def plot_quantile_fig(data_,path,axis=-1):
    '''
    axis:需要查看的数据维度，保留的数据维度
    '''
    # width = 1
    torch.cuda.empty_cache()
    height = len(data_)
    fig,axes = plt.subplots(1,1)
    torch.cuda.synchronize(),torch.cuda.empty_cache()

    if isinstance(data_,torch.Tensor):
        if data_.requires_grad:
            data = data_.detach().cpu().numpy()
        else:
            data = data_.cpu().numpy()
    shape = data.shape
    if axis >= len(shape):
        raise ValueError("Axis should be less than data.shape")
    permuted_data = np.moveaxis(data, axis, 0)
    reshaped_data = permuted_data.reshape(shape[axis], -1).transpose(1,0)
    reshaped_data = torch.from_numpy(reshaped_data).cpu().float()
    pmax = torch.amax(reshaped_data,dim=0).cpu().numpy()
    p9999 = torch.quantile(reshaped_data,0.9999,dim=0).cpu().numpy()
    p99 = torch.quantile(reshaped_data,0.99,dim=0).cpu().numpy()
    p75 = torch.quantile(reshaped_data,0.75,dim=0).cpu().numpy()
    p25 = torch.quantile(reshaped_data,0.25,dim=0).cpu().numpy()
    p01 = torch.quantile(reshaped_data,0.01,dim=0).cpu().numpy()
    p0001 = torch.quantile(reshaped_data,0.0001,dim=0).cpu().numpy()
    pmin = torch.amin(reshaped_data,dim=0).cpu().numpy()
    x_label_ids = np.arange(len(pmin))
    del reshaped_data
    torch.cuda.synchronize(),torch.cuda.empty_cache()
    axes.plot(x_label_ids,p9999,color='red',label='1/9999 Percentile',linewidth=0.5)
    axes.plot(x_label_ids,p99,color='purple',label='1/99 Percentile',linewidth=0.5)
    axes.plot(x_label_ids,p75,color='orange',label='25/75 Percentile',linewidth=0.5)
    axes.plot(x_label_ids,p25,color='orange',linewidth=0.5)
    axes.plot(x_label_ids,p01,color='purple',linewidth=0.5)
    axes.plot(x_label_ids,p0001,color='red',linewidth=0.5)
    axes.plot(x_label_ids,pmax,color='blue',linewidth=0.5)
    axes.plot(x_label_ids,pmin,color='blue',label='Min/Max',linewidth=0.5)

    axes.set_xlabel('Hidden dimension index')
    axes.set_ylabel('Activation value')
    axes.legend(loc='upper right')

    fig.tight_layout(rect=[0,0.05,1,0.95])
    fig.savefig(path,dpi=300)
    print("saveing: ",path)
    plt.close()


def plot_box_data_perchannel_fig(data_,path,axis=-1):
    '''
    axis:需要查看的维度，保留的维度
    '''
    if isinstance(data_,torch.Tensor):
        if data_.requires_grad:
            data = data_.detach().cpu().numpy()
        else:
            data = data_.cpu().numpy()
    if len(data.shape)==3:
        data = np.max(data,axis=0)
    shape = data.shape
    if axis >= len(shape):
        raise ValueError("Axis should be less than data.shape")
    permuted_data = np.moveaxis(data, axis, 0)
    reshaped_data = permuted_data.reshape(shape[axis], -1)
    plt.figure(figsize=(max(shape[axis] // 10, 6),6))
    plt.title(path)
    plt.boxplot(reshaped_data.T)
    plt.xticks(range(0,shape[axis]+1,10))
    try:
        plt.tight_layout()
    except Exception as e:
        print(f"Warning: tight_layout failed with error: {e}")
    plt.savefig(path)
    plt.close()
    print('saving:  ', path)

def plot_bar_fig(data_,path):
    if isinstance(data_,torch.Tensor):
        if data_.requires_grad:
            data = data_.detach().cpu().numpy()
        else:
            data = data_.cpu().numpy()
    data_range = np.max(data) - np.min(data)
    bin_width = data_range / 30  # 设定每个区间的宽度
    bins = np.arange(np.min(data), np.max(data) + bin_width, bin_width)
    plt.hist(data.reshape(-1), bins=bins, edgecolor='black')
    plt.title(path)
    plt.xlabel("x_val")
    plt.ylabel("num")
    try:
        plt.tight_layout()
    except Exception as e:
        print(f"Warning: tight_layout failed with error: {e}") #用于自动调整子图参数，使图形中的子图、标签、标题等不重叠，并且整体布局更加紧凑美观
    plt.savefig(path)
    plt.close()
    print('saving:  ', path)

def plot_bar3d_fig(data_,path,axis=-1):
    '''
    data_:消除了batch维度的数据,在最后一维度上展开数据
    
    '''
    plt.ioff()
    if isinstance(data_,torch.Tensor):
        data_ = data_.reshape(data_.shape[0],-1).abs()
        if data_.requires_grad:
            data = data_.detach().cpu().numpy()
        else:
            data = data_.cpu().numpy()
    else:
        data = data_
        
    # shape = data.shape
    # if axis >= len(shape):
    #     raise ValueError("Axis should be less than data.shape")
    # permuted_data = np.moveaxis(data, axis, 0)
    # reshaped_data = permuted_data.reshape(shape[axis], -1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 获取数据的维度
    x_len, y_len = data.shape

    # 生成X和Y的坐标
    _x = np.arange(x_len)
    _y = np.arange(y_len)
    
    # meshgrid的参数顺序应该与数据存储的顺序一致：行 -> x轴, 列 -> y轴
    _xx, _yy = np.meshgrid(_x, _y, indexing="ij")
    x, y = _xx.ravel(), _yy.ravel()

    # 数据展开为一维
    top = data.ravel()
    bottom = np.zeros_like(top)
    
    # 调整宽度和深度
    width = depth = 0.2

    # 使用颜色映射工具
    colors = plt.cm.viridis(top / float(top.max()))

    # 绘制3D柱状图
    ax.bar3d(x, y, bottom, width, depth, top, shade=True, color=colors)

    # 设置轴标签和标题
    plt.title(path)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_zlabel("Value")
    
    # 添加颜色条
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(top)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    
    # 调整视角
    ax.view_init(elev=30, azim=45)

    # try:
    #     plt.tight_layout()
    # except Exception as e:
    #     print(f"Warning: tight_layout failed with error: {e}")
        
    # 保存图像
    plt.savefig(path)
    plt.close()
    print('saving:  ', path)

def plot_bar3d_fig_1(data_, path):#plotly实现
    import plotly.graph_objects as go
    
    if isinstance(data_, torch.Tensor):
        data_ = data_.reshape(data_.shape[0], -1).abs()
        if data_.requires_grad:
            data = data_.detach().cpu().numpy()
        else:
            data = data_.cpu().numpy()
    else:
        data = data_

    # 获取数据的维度
    x_len, y_len = data.shape

    # 生成X和Y的坐标
    _x = np.arange(x_len)
    _y = np.arange(y_len)
    
    # meshgrid的参数顺序应该与数据存储的顺序一致：行 -> x轴, 列 -> y轴
    _xx, _yy = np.meshgrid(_x, _y, indexing="ij")
    x, y = _xx.ravel(), _yy.ravel()

    # 数据展开为一维
    top = data.ravel()

    # 初始化空的 Figure 对象
    fig = go.Figure()

    # 获取颜色渐变
    max_top = top.max()
    colors = plt.cm.viridis(top / max_top)

    # 添加柱体从底部到顶部的线条，并使用颜色渐变
    for i in range(len(x)):
        fig.add_trace(go.Scatter3d(
            x=[x[i], x[i]],
            y=[y[i], y[i]],
            z=[0, top[i]],  # 从底部（z=0）到顶部（z=top[i]）
            mode='lines',
            line=dict(color=f'rgb({colors[i][0]*255},{colors[i][1]*255},{colors[i][2]*255})', width=2)
        ))

    # 设置轴标签和标题
    fig.update_layout(
        title=path,
        scene=dict(
            xaxis=dict(title='Column'),
            yaxis=dict(title='Row'),
            zaxis=dict(title='Value'),
        )
    )

    # 保存图像为 PNG 文件
    fig.write_html(path)
    print('saving:  ', path)
  
def find_images(directory, prefix, suffix):
    """
    在指定目录中查找符合特定前缀和后缀的图片文件。
    
    参数:
    - directory: 图片所在的目录。
    - prefix: 文件名前缀。
    - suffix: 文件名后缀。
    
    返回:
    - 符合条件的图片文件路径列表。
    """
    files = os.listdir(directory)
    selected_files = [os.path.join(directory, f) for f in files if f.startswith(prefix) and f.endswith(suffix)]
    return selected_files

def concat_images(image_paths, images_per_row, save_path):
    """
    将多张图片拼接成一张大图。

    参数:
    - image_paths: 图片路径的列表。
    - images_per_row: 每行图片的数量。
    - save_path: 拼接后的图片保存路径。
    """
    if not image_paths:
        print("没有找到符合条件的图片。")
        return
    
    # 加载第一张图片以获取单张图片的尺寸
    with Image.open(image_paths[0]) as img:
        img_width, img_height = img.size
    
    # 计算大图的尺寸
    rows = (len(image_paths) + images_per_row - 1) // images_per_row
    concat_image = Image.new('RGB', (images_per_row * img_width, rows * img_height))
    
    # 按顺序拼接图片
    for idx, img_path in enumerate(image_paths):
        row = idx // images_per_row
        col = idx % images_per_row
        with Image.open(img_path) as img:
            concat_image.paste(img, (col * img_width, row * img_height))
    
    # 保存拼接后的图片
    try:
        concat_image.save(save_path)
        print(f"图片已拼接并保存到 {save_path}")
    except Exception as e:
        print(f"保存拼接后的图片时出错: {e}")

def count_suffixes(directory):
    """
    统计指定文件夹中图片文件名中'mixer.'后面部分的后缀名称及其数量。
    
    参数:
    - directory: 包含图片的文件夹路径。
    
    返回:
    - 一个字典，键为后缀名称，值为该后缀名称出现的次数。
    """
    suffix_counts = {}
    for filename in os.listdir(directory):
        # 分割文件名以找到'mixer.'后面的部分
        parts = filename.split('mixer.')
        if len(parts) > 1:
            suffix = parts[-1]  # 获取'mixer.'后面的部分
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    return suffix_counts

def find_and_cat_figs_from_blocks(path):
    if os.path.exists(path):
        suffixes = count_suffixes(path)
        for ss in suffixes.keys():
            image_paths = find_images(path, "", ss)
            filter_images = [image for image in image_paths if image.split(".")[0].split("/")[-1].isdigit()]
            sorted_images = sorted(filter_images,key=lambda x: int(x.split(".")[0].split("/")[-1]))
            os.makedirs(path+"_cat/", exist_ok=True)
            concat_images(sorted_images, 6, path+f"_cat/{ss}")

def delete_files_with_special_name(name,directory):
    """
    删除某个目录下名称包含 "conv" 的所有文件。
    
    参数:
    directory (str): 要删除文件的目录路径。
    """
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        # 检查文件名是否包含 "conv"
        if name in filename:
            # 构建完整的文件路径
            file_path = os.path.join(directory, filename)
            # 删除文件
            os.remove(file_path)
            print(f"已删除文件: {filename}")

if __name__ == "__main__":
    
    a = torch.randn(10, 10)
    # time_1 = time.time()
    # plot_bar3d_fig(a,'/data01/home/xuzk/workspace/mamba_quant_comp/model_llm/data/tmp.jpg')
    # time_2 = time.time()
    # plot_bar3d_fig_1(a,'/data01/home/xuzk/workspace/mamba_quant_comp/model_llm/data/tmp_1.jpg')
    # time_3 = time.time()
    # print(f"matplotlib程序执行时间为：{time_2-time_1} 秒")
    # print(f"other's 程序执行时间为：{time_3-time_2} 秒")
    
    # delete_files_with_special_name("bar_data","/data01/home/xuzk/workspace/mamba_quant_comp/model_llm/data/analyse_fig/fig/fp_data")
    
    find_and_cat_figs_from_blocks('/data01/home/xuzk/workspace/mamba_quant_comp/model_vim_quant/data/analyse_fig/fig_after_r1r2r3r5r6_k1k5_base/fp_data')
    
    pass

