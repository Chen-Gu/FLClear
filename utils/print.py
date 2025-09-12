import matplotlib.pyplot as plt

'''嵌入水印和未嵌入水印的准确度画图'''
file_with_1 = '../result/VGG13_None.txt'
file_with_2 = '../result/VGG13_base.txt'

# 读取文件并解析数据
def wm_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            epoch = int(parts[0].split(' ')[1])
            print(epoch)
            ssim = float(parts[1].split(':')[1].strip())
            test_accuracy = float(parts[2].split(':')[1].strip()[:-1])
            data.append({'epoch': epoch, 'ssim': ssim, 'test_accuracy': test_accuracy})
    return data
def nowm_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            epoch = int(parts[0].split(' ')[1])
            test_accuracy = float(parts[1].split(':')[1].strip()[:-1])
            data.append({'epoch': epoch, 'test_accuracy': test_accuracy})
    return data
# 解析两个文件的数据
'''
data_with_2 = no_wm_file(file_with_2)
data_with_1 = wm_file(file_with_1)

# 提取数据
epochs = [entry['epoch'] for entry in data_with_1]  # 假设两个文件的epoch是相同的
test_acc = [entry['test_accuracy'] for entry in data_with_1]
ssim = [entry['bn_ssim'] for entry in data_with_1]
acc = [entry['test_accuracy'] for entry in data_with_2]
# 绘制折线图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, test_acc, label='Test Accuracy with wm', marker='o')
plt.plot(epochs, acc, label='Test Accuracy', marker='o')
last_acc_wm = test_acc[-1]
last_acc = acc[-1]
plt.axhline(y=last_acc_wm, color='blue', linestyle='--', alpha=0.5, label=f'Last acc with wm: {last_acc_wm:.2f}%')
plt.axhline(y=last_acc, color='orange', linestyle='dashdot', alpha=0.5, label=f'Last acc: {last_acc:.2f}%')
# 在 Y 轴上标注最后一个数据点的 Y 值
if last_acc_wm > last_acc:
    plt.annotate(f'{last_acc_wm:.2f}%',
                 xy=(0, last_acc_wm),  # 标注点的位置（Y 轴上）
                 xytext=(-30, 0),  # 文本偏移（向左 30 单位）
                 textcoords='offset points',
                 ha='right', va='center',
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'),
                 arrowprops=dict(arrowstyle='->'))
else:
    plt.annotate(f'{last_acc:.2f}%',
                 xy=(0, last_acc),  # 标注点的位置（Y 轴上）
                 xytext=(-30, 0),  # 文本偏移（向左 30 单位）
                 textcoords='offset points',
                 ha='right', va='center',
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'),
                 arrowprops=dict(arrowstyle='->'))
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('IID VGG13')
# 显示图表
# 绘制 SSIM 折线图
plt.subplot(1, 2, 2)
plt.plot(epochs, ssim, label='SSIM', marker='*')
last_s = ssim[-1]
plt.axhline(y=last_s, color='g', linestyle='--', alpha=0.5, label=f'Last S: {last_s:.2f}')
# 在 Y 轴上标注最后一个数据点的 Y 值
plt.annotate(f'{last_s:.2f}',
             xy=(0, last_s),  # 标注点的位置（Y 轴上）
             xytext=(-30, 0),  # 文本偏移（向左 30 单位）
             textcoords='offset points',
             ha='right', va='center',
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'),
             arrowprops=dict(arrowstyle='->'))
plt.xlabel('Epoch')
plt.ylabel('SSIM Score')
plt.legend()
plt.grid(True)
plt.show()
'''

file = '../result/AlexNet/C10_FedAvg_0.0_l1.txt'
file1 = '../result/AlexNet/C10_FedAvg_0.5_l1.txt'
file2 = '../result/AlexNet/C10_FedAvg_1.0_l1.txt'
data = wm_file(file)
data1 = wm_file(file1)
data2 = wm_file(file2)

epochs = [entry['epoch'] for entry in data]
acc = [entry['test_accuracy'] for entry in data]
acc1 = [entry['test_accuracy'] for entry in data1]
acc2 = [entry['test_accuracy'] for entry in data2]
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label='0.0l1')
plt.plot(epochs, acc1, label='0.5l1')
plt.plot(epochs, acc2, label='1.0l1')
plt.legend()
plt.grid(True)
ssim = [entry['ssim'] for entry in data]
ssim1 = [entry['ssim'] for entry in data1]
ssim2 = [entry['ssim'] for entry in data2]
plt.subplot(1, 2, 2)
plt.plot(epochs, ssim, label='0.0l1')
plt.plot(epochs, ssim1, label='0.5l1')
plt.plot(epochs, ssim2, label='1.0l1')
plt.title(f"l1")
plt.legend()
plt.grid(True)
plt.savefig(f"../result/AlexNet/img/l1_80.png")
plt.close()
# plt.show()