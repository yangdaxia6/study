import torch

torch.manual_seed(20)
def center_loss():
    totalClass=3
    feature_dim=2
    delta4Center=0.1
    centers = torch.tensor([1, 2, 3, 1.1, 2.1, 3.1]).view(feature_dim, totalClass).float()
    #centers = torch.randint(0, 2, (feature_dim, totalClass)).float()
    import pdb
    pdb.set_trace()
    centers_inter = centers.repeat(1, totalClass).view(feature_dim, totalClass, totalClass)#(64, 3, 3)
    centers_self_rep = centers.repeat(totalClass, 1).permute(1, 0).view(totalClass, totalClass, feature_dim).permute(2, 0, 1)#(64, 3, 3)
    center_diff = torch.add(centers_self_rep, -1, centers_inter)#(64, 3, 3)
    center_diff = torch.norm(center_diff, dim=0)#(3, 3)
    center_diff = torch.add(center_diff, -1 * delta4Center)
    center_diff = torch.add(center_diff, -1, torch.diag(torch.diagonal(center_diff)))
    #center_diff = torch.clamp(center_diff, -999, 0)
    center_loss = torch.pow(center_diff, 2).sum() / (totalClass * (totalClass - 1) + 1e-7)
    print(center_loss)

def center_loss_2():
    totalClass=3
    feature_dim=64
    delta4Center=0.1
    centers = torch.randn(feature_dim, totalClass)
    center_inter = centers.repeat(1, totalClass)

def repeat():
     a = torch.tensor([1, 2, 3, 1.1, 2.1, 3.1]).view(2, 3)
     print(a)
     print(a.repeat(1, 3))
     print(a.repeat(0, 3))


def view():
    B, C, H, W = 2, 4, 3, 6
    a = torch.randn(B*C)
    aa = torch.randn(B, C*H)
    b = a.view(B, C)
    c = aa.view(B, C, H)
    #c = a.repeat(num, 1).permute(1, 0).view(num, num, 2)
    print(a, '\n\n', b, '\n\n', c)

def repeat():
    num = 3
    a = torch.randn(2, num)
    b = a.repeat(1, num).view(2, num, num)
    c = a.repeat(num, 1).permute(1, 0).view(num, num, 2).permute(2, 0, 1)
    print(a, '\n\n', b, '\n\n', c)

if __name__=='__main__':
    #repeat()
    #view()
    #center_loss()
    repeat()