import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ultralytics.nn.modules.conv import Conv

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class Arcface(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, embedding_size=1280, classnum=2,  s=64., m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings,kernel_norm)
#         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, embedding_size=1280, classnum=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = classnum
        self.feat_dim = embedding_size
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        
        return loss    

   
class ArcFaceHead(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):

         """Initializes YOLOv8 classification head with specified input and output channels, kernel size, stride,
         padding, and groups.
         """
         super(ArcFaceHead, self).__init__()
         hidden_channels=1280
         self.classifier = nn.Linear(hidden_channels, c2)

    def forward(self, x):
         """Performs a forward pass of the YOLO model on input image data."""
         feature=x
         x = self.classifier(x)
         
         return torch.cat((x, feature), dim=1) if self.training else x.softmax(1)
     

class ArcFaceLoss:
    """Criterion class for computing training losses."""
    def __init__(self,hidden_channels,nc,s,m,training) -> None:
        self.criterion = Arcface(embedding_size=hidden_channels,classnum=nc,s=s,m=m).cuda()
        self.training=training
        self.nc=nc
    
    def __call__(self, preds, batch):
        if self.training:
            shape=preds.shape[1]
            cls,feature=preds[:,:shape-self.nc],preds[:,:shape-self.nc:]
            output=self.criterion(feature,batch['cls'])
            
            loss = torch.nn.functional.cross_entropy(output, batch["cls"], reduction="mean")+torch.nn.functional.cross_entropy(cls, batch["cls"], reduction="mean")
            loss_items = loss.detach()
            return loss.sum(), loss_items
        else:
            loss = torch.nn.functional.cross_entropy(preds, batch["cls"], reduction="mean")
            loss_items = loss.detach()
            return loss, loss_items
        
class ArcFace_Center_Loss:
    """Criterion class for computing training losses."""
    def __init__(self,hidden_channels,nc,s,m,training,l1,l2) -> None:
        self.criterion_arcface = Arcface(embedding_size=hidden_channels,classnum=nc,s=s,m=m).cuda()
        self.criterion_centerloss = CenterLoss(embedding_size=hidden_channels,classnum=nc).cuda()
        self.training=training
        self.nc=nc
        self.l1=l1
        self.l2=l2
    
    def __call__(self, preds, batch):
        if self.training:
            shape=preds.shape[1]
            cls,feature=preds[:,:shape-self.nc],preds[:,:shape-self.nc:]
            output_arcface=self.criterion_arcface(feature,batch['cls'])
            output_centerloss=self.criterion_centerloss(feature,batch['cls'])
            
            loss = self.l1*torch.nn.functional.cross_entropy(output_arcface, batch["cls"], reduction="mean")+self.l2*output_centerloss+torch.nn.functional.cross_entropy(cls, batch["cls"], reduction="mean")
            loss_items = loss.detach()
            return loss.sum(), loss_items
        else:
            loss = torch.nn.functional.cross_entropy(preds, batch["cls"], reduction="mean")
            loss_items = loss.detach()
            return loss, loss_items