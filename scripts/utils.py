from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

def feature_vis(logstic, epoch):
    embs = logstic.detach().cpu().numpy()
    embs = embs.reshape(-1, 512)
    print("vis clip feature")
    tsne = TSNE(n_components=2, learning_rate=200, metric='cosine', random_state=1)
    tsne.fit_transform(embs)
    outs_2d = np.array(tsne.embedding_)
    
    css4 = list(mcolors.CSS4_COLORS.keys())
    #我选择了一些较清楚的颜色，更多的类时也能画清晰
    color_ind = [2,7,9,10,11,13,14,16,17,19,20,21,25,28,30,31,32,37,38,40,47,51,
            55,60,65,82,85,88,106,110,115,118,120,125,131,135,139,142,146,147]
    css4 = [css4[v] for v in color_ind]
    
    id_ = [0, 66, 128, 256, 387, 576, 632, 765, 876, 987]
    i = 0
    for lbi in id_:
        temp = outs_2d[lbi*100: (lbi+1)*100]
        plt.plot(temp[:,0],temp[:,1],'.',color=css4[i])
        i += 1
    plt.title('feats dimensionality reduction visualization by tSNE,test data')
    plt.savefig(f'feature_epoch={epoch}.png')
    plt.clf()

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, start_lr, total_epoch, after_scheduler=None):
        self.start_lr = start_lr
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.total_epoch:
            return [self.start_lr + (base_lr - self.start_lr) * self.last_epoch / (self.total_epoch - 1) for base_lr in self.base_lrs]
        elif self.after_scheduler:
            if not self.finished:
                # 重要：在预热结束后，第一次调用余弦退火调度器
                self.after_scheduler.step(self.total_epoch)
                self.finished = True
            # 正常调用余弦退火调度器的 get_lr 方法
            return self.after_scheduler.get_lr()
        return self.base_lrs