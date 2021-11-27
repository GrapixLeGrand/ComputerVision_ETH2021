import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        # TODO
        
        self.layers = nn.Sequential(
            
            #layer 1
            nn.Conv2d(3, 8, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            #layer 2
            nn.Conv2d(8, 8, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            #layer 3
            nn.Conv2d(8, 16, (5, 5), stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            #layer 4
            nn.Conv2d(16, 16, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            #layer 5
            nn.Conv2d(16, 16, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            #layer 6
            nn.Conv2d(16, 32, (5, 5), stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            #layer 7
            nn.Conv2d(32, 32, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            #layer 8
            nn.Conv2d(32, 32, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            #layer 9
            nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)
        )

    def forward(self, x):
        # x: [B,3,H,W] # B, CHAnnels, Height, Width
        # TODO
        return self.layers(x)


class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        None
        # TODO

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO
        None


#src_fea or source_features is the output of the neural network
# that searched the features (With C channels)
def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B,C,H,W = src_fea.size()
    D = depth_values.size(1)
    # compute the warped positions with depth values
    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)

        warped_src_fea = rot @ torch.inverse(ref_proj)

        None
        # TODO

    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    # TODO

    return warped_src_fea

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    # TODO
    None

def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO
    None

def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO
    None
