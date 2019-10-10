iimport torch
import torch.nn as nn

#Semantic Instance Segmentation with a Discriminative Loss Function
class NewPairLoss(nn.Module):
    def __init__(self,delta4Center,delta4Point,para_var,para_dis):
        super(NewPairLoss,self).__init__()
        self.delta4Center = delta4Center
        self.delta4Point = delta4Point
        self.para_var = para_var
        self.para_dis = para_dis

    def forward(self,x,y):
        # x [B, C, H, W]: the instance output of the model 
        # y [B, H, W]: the instance ground truth, with different IDs for different instances, such as [0, 1, 2] 
        batchSize, feature_dim,_ ,_ = x.shape
        losses = []
        c_losses = []
        p_losses = []
        for batch in range(batchSize):
            pred_feature = x[batch].view(feature_dim,-1)
            #pred_feature = torch.div(pred_feature,torch.norm(pred_feature,dim = 0))
            
            gt_label = y[batch].view(-1)
            ##calculate the center of each class
            totalClass = torch.max(gt_label).item() + 1
            centers = torch.zeros(feature_dim,totalClass).cuda()
            cnt = torch.bincount(gt_label).float().cuda()
            #for i in range(gt_label.shape[0]):
            #    centers[:,gt_label[i]] += pred_feature[:,i]
            #    cnt[gt_label[i]] += 1
            for i in range(totalClass):
                if (pred_feature[:,gt_label == i].shape[0] == 0):
                    continue
                centers[:,i] = pred_feature[:,gt_label == i].sum(dim = 1)
                
            centers = torch.div(centers,cnt + 1e-7)
            #centers = torch.div(centers,torch.norm(centers,dim = 0) + 1e-7)  
            center_expand = torch.index_select(centers,1,gt_label)
            ##calculate l_var

            dist = torch.norm(torch.add(center_expand,-1,pred_feature),dim = 0)
            dist = torch.add(dist, -1 * self.delta4Point)
            dist = torch.clamp(dist,0,999)
            dist = torch.pow(dist,2)
            point_loss = torch.zeros(totalClass).cuda()
            #for i in range(gt_label.shape[0]):
            #    point_loss[gt_label[i]] += dist[i]
            for i in range(totalClass):
                point_loss[i] = dist[gt_label == i].sum()
            point_loss = torch.div(point_loss,cnt + 1e-7)
            point_loss = point_loss.sum() / totalClass
            #print(centers)
            ##calculate l_dist
            centers_inter = centers.repeat(1,totalClass).view(feature_dim,totalClass,totalClass)
            #centers_self_rep = centers.repeat(totalClass,1).permute(1,0).view(totalClass,totalClass,feature_dim).permute(2,0,1)
            centers_self_rep = centers.permute(0, 2, 1)
            center_diff = torch.add(centers_self_rep,-1,centers_inter)
            center_diff = torch.norm(center_diff,dim = 0)
            center_diff = torch.add(center_diff,-1 * self.delta4Center)
            center_diff = torch.add(center_diff,-1,torch.diag(torch.diagonal(center_diff)))
            center_diff = torch.clamp(center_diff,-999,0)
            center_loss = torch.pow(center_diff,2).sum() / (totalClass * (totalClass - 1) + 1e-7)
            point_loss = self.para_var * point_loss
            center_loss = self.para_dis * center_loss
            loss = point_loss + center_loss
            losses.append(loss)
            c_losses.append(center_loss)
            p_losses.append(point_loss)
        loss = torch.stack(losses)
        c_loss = torch.stack(c_losses)
        p_loss = torch.stack(p_losses)
        return loss.sum(),c_loss.sum(),p_loss.sum()

