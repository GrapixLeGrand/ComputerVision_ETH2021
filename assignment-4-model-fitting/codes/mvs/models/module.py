import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        x = x.to(device)
        return self.layers(x)


class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()

        # TODO
        # in: S, out: C0
        self.L1 = nn.Sequential(
            nn.Conv2d(G, 8, (3, 3), stride=1, padding=1),
            nn.ReLU()
        )

        # in: C0, out: C1
        self.L2 = nn.Sequential(
            nn.Conv2d(8, 16, (3, 3), stride=2, padding=1),
            nn.ReLU()
        )

        # in: C1, out: C2
        self.L3 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3), stride=2, padding=1),
            nn.ReLU()
        )

        # in: C2, out: C3
        self.L4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, (3, 3), stride=2, padding=1, output_padding=1),
        )

        # in: C3 + C1, out: C4
        self.L5 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, (3, 3), stride=2, padding=1, output_padding=1),
        )

        # in: C4 + C0, out: C5
        self.L6 = nn.Sequential(
            nn.Conv2d(8, 1, (3, 3), stride=1, padding=1),
        )

    # https://medium.com/analytics-vidhya/pytorch-contiguous-vs-non-contiguous-tensor-view-understanding-view-reshape-73e10cdfa0dd
    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO  
        
        x = x.to(device)
        
        B, G, D, H, W = x.size()
        S = x.contiguous()
        S = S.view(B * D, G, H, W) # correct view ? I dont think so.
        
        C0 = self.L1(S)
        C1 = self.L2(C0)
        C2 = self.L3(C1)
        C3 = self.L4(C2)
        C4 = self.L5(C3 + C1)
        C5 = self.L6(C4 + C0)

        C5 = C5.reshape((B, D, H, W)) # warning

        return C5

#src_fea or source_features is the output of the neural network
# that searched the features (With C channels and B images)
def warping(src_fea, src_proj, ref_proj, depth_values):
    
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]

    src_fea = src_fea.to(device)
    src_proj = src_proj.to(device)
    ref_proj = ref_proj.to(device)
    depth_values = depth_values.to(device)

    B,C,H,W = src_fea.size()
    D = depth_values.size(1)
    
    # In this function we are given a block of 32 images coming from the
    # neural network. The transformation of the sources (B one as we
    # need one per image) and the reference transformation. We are also
    # given some depth values on the reference frame. 

    # What we want is to rely these depth values for some pixels in the
    # reference frame to our sources frames.

    xyz = torch.zeros((B, H, W, 2)).to(device)
    
    xyz.requires_grad = False # Warning
    warped_src_fea = torch.zeros((B, C, D, H, W)).to(device) #src_fea.detach().clone()
    
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
            y, x = y.to(device), x.to(device)
            # TODO

            ones = torch.ones(y.shape).to(device)
            yx1 = torch.stack((y, x, ones), dim=1).to(device) #stack xyz as vector => axis = 1
            yx1 = torch.reshape(yx1, (H, W, 3)).to(device)
            yx1_copy = yx1.detach().clone().to(device)
            #stacked = torch.cat(B * [yx1.unsqueeze(0)])
           
            for j in range(B):

                current_depth = depth_values[j][i]
                yx1_copy *= current_depth
                test = rot[j]
                test_t = rot[j].T

                #yx1_copy = torch.einsum("hwk,ij->hwk", yx1_copy, rot[j]) + trans[j].T
                yx1_copy = yx1_copy @ rot[j].T + trans[j].T # warning rot T, is this this and does T works?
                yx1_copy[:, :, 0] /= yx1_copy[:, :, -1]
                yx1_copy[:, :, 1] /= yx1_copy[:, :, -1] # projecting back to the source image plane
                xyz[j, :, :, :] = yx1_copy[:, :, : -1]

            warped_src_fea[:, :, j, :, :] = torch.nn.functional.grid_sample(src_fea, xyz)

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

    ref_fea = ref_fea.to(device)
    warped_src_fea = warped_src_fea.to(device) 

    B, C, D, H, W = warped_src_fea.size()
    out = torch.zeros((B, G, D, H, W))
    group_size = C // G
    factor = 1.0 / (C / G)

    for g in range(G):
        l = group_size * g
        u = group_size * (g + 1)
        # not quite sure lol xD
        out[:, g, :, :, :] = factor * torch.einsum("bchw,bcdhw->", ref_fea[:, l:u, :, :], warped_src_fea[:, l:u, :, :, :])

    return out

def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO
    p = p.to(device)
    result = torch.einsum("bd,bdhw->bhw", depth_values[:, :], p[:, :, :, :])
    return result

def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO

    
    
    #mask_u = mask.unsqueeze(1)
    #depth_gt_t = depth_gt_t.unsqueeze(1)
    #depth_est_u = depth_est.unsqueeze(1)

    #mask_u = mask_u > 0 #convert to boolean in order to avoid indice extraction
    mask = mask > 0
    est_masked = depth_est[mask] #.view((B, H, W)) #depth_est_u[mask_u]
    gt_masked = depth_gt[mask] #.view((B, H, W)) #depth_gt_t[mask_u]

    loss = torch.nn.L1Loss()(est_masked, gt_masked)

    return loss

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