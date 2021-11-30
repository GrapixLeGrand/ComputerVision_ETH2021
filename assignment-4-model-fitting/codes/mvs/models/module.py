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
# that searched the features (With C channels and B images)
def warping(src_fea, src_proj, ref_proj, depth_values):
    
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B,C,H,W = src_fea.size()
    D = depth_values.size(1)
    
    # In this function we are given a block of 32 images coming from the
    # neural network. The transformation of the sources (B one as we
    # need one per image) and the reference transformation. We are also
    # given some depth values on the reference frame. 

    # What we want is to rely these depth values for some pixels in the
    # reference frame to our sources frames.

    xyz = torch.zeros((B, H, W, 2))
    
    xyz.requires_grad = False # Warning
    warped_src_fea = src_fea.detach().clone()

    for i in range(D):
        
        xyz[:, :, :, :] = 0.0
        # compute the warped positions with depth values
        with torch.no_grad():
            # relative transformation from reference to source view (for each images)
            proj = torch.matmul(src_proj, torch.inverse(ref_proj))
            rot = proj[:, :3, :3].float()  # [B,3,3]
            trans = proj[:, :3, 3:4].float()  # [B,3,1]

            # x and y are the pixels of our data images
            y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                                torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
            # as doc says it gives the elements contiguous in memory (change result ? i think no, this is just for more efficient memory accesses after a permutation)
            y, x = y.contiguous(), x.contiguous()
            # View in 1D => flatten the 2D arrays
            y, x = y.view(H * W), x.view(H * W) # I guess these are the x, y coordinates of pixels
            
            # TODO

            ones = torch.ones(y.shape)
            yx1 = torch.stack((y, x, ones), dim=1) #stack xyz as vector => axis = 1
            yx1 = torch.reshape(yx1, (H, W, 3))
            yx1_copy = yx1.detach().clone()
            #stacked = torch.cat(B * [yx1.unsqueeze(0)])

            for j in range(B):

                current_depth = depth_values[j][i]
                yx1_copy *= current_depth
                yx1_copy =  yx1_copy @ rot[j] + trans[j].T # warning rot T, is this this and does T works?
                yx1_copy[:, :, 0] /= yx1_copy[:, :, -1]
                yx1_copy[:, :, 1] /= yx1_copy[:, :, -1] # projecting back to the source image plane
                xyz[j, :, :, :] = yx1_copy[:, :, : -1]

            warped_src_fea = torch.nn.functional.grid_sample(warped_src_fea, xyz)
            

    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    # TODO
    #a = src_fea[0]
    #warped_src_fea = torch.nn.functional.grid_sample(src_fea, xyz)

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

"""
#yx1_piled = torch.cat(2 * [yx1.unsqueeze(0)]) # unsqueeze one slice and pile them D times (one for each depth sample) #yx1.unsqueeze(0).repeat()

            #rot @ (depth_values) + trans
            #xyz = yx1_piled #= torch.cat(B * [yx1_piled.unsqueeze(0)])
            #warped_src_fea = rot @ torch.inverse(ref_proj)
"""
"""
        ones = torch.ones(y.shape)
        yx1 = torch.stack((y, x, ones), dim=1) #stack xyz as vector => axis = 1
        
        #print(yx1[0])
        #print(yx1[1])
        #print(yx1[2])

        yx1 = torch.reshape(yx1, (H, W, 3))
        yx1_piled = torch.cat(D * [yx1.unsqueeze(0)]) # unsqueeze one slice and pile them D times (one for each depth sample) #yx1.unsqueeze(0).repeat()
        
        depth_repeated = depth_values.repeat_interleave(D * 3 * H * W, dim=1).reshape((B, H, W, 3))
        
        a = torch.tensor([
            [1, 2, 3],
            [4, 5, 6]
        ])

        a = a.repeat_interleave(3 * H * W, dim=1)

        print(a)
        print(a.size())

        a = torch.tensor([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [4, 10, 6], [13, 8, 9]]
        ])


        b = a * a
        print(b)
        c = torch.matmul(a, b)

        b = a * 3
        c = a * torch.tensor([3]).squeeze(0)
        d = torch.tensor([3])
        d = torch.tensor([3]).squeeze(0)

        #un_packed = depth_values[0]
        #yx1_piled_images = yx1_piled_images * depth_values
        
        #roti = rot[0].unsqueeze(0)
        i = rot[0]
        a = yx1 @ rot[0].T

        rot @ (depth_values) + trans

        #warped_src_fea = rot @ torch.inverse(ref_proj)
"""