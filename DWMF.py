import numpy as np
import torchvision
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

#############################################################
# @DeepFool with watermask（logo）
# multiclass
#############################################################
class DeepWaterMarkFool(object):
    def __init__(self, num_classes, net):
        self.num_classes = num_classes                          # number of classes
        self.net = net                                          # network to be attacked (VGG)
        self.softmax = torch.nn.Softmax(dim=1)                  # softmax : Calculate the confidence
        self.transform4VGG = transforms.Compose([               # Resize and crop image to 224x224
            transforms.Resize((256, 224)),
            transforms.CenterCrop(224)
        ])
        self.transform4wm = transforms.Resize(80)              # Resize watermark logo to 80x80
        self.mean_bgr = np.array([129.1863, 104.7624, 93.5940])
        self.mean_rbg = np.array([93.5940, 104.7624, 129.1863])
        self.cuda = torch.cuda.is_available()                  
        if self.cuda:
            net = net.cuda()

    def AddingWaterMark(self, bkg_path, wm_path, start_point=(224-80,0)):
        bkg_img = Image.open(bkg_path)
        bkg_img = self.transform4VGG(bkg_img)               # <Image>Size(224,224)
        img1 = np.array(bkg_img).astype(np.float64)         # -><numpy array>[H W C]
        
        img1 = img1[:, :, ::-1]                             # Change RGB to BGR -[H W C]
        img1 -= self.mean_bgr                               # minus channel wise mean(BGR)
        img1 = img1.transpose((2, 0, 1))                    # [H W C]->[C H W]
        img1 = torch.from_numpy(img1.copy()).float()
        img1 = img1.cuda() if self.cuda else img1
        img1 = img1.unsqueeze(0) if len(img1.size()) == 3 else img1     

        # bkgPlusWM 
        wm_img = Image.open(wm_path)                                    # <Image>
        
        wm_img = self.transform4wm(wm_img)                              # Resize Watermark

        layer = Image.new('RGBA', bkg_img.size, (0, 0, 0, 0))           
        layer.paste(wm_img, start_point)                                
        r, g, b, a = layer.split()                                     
        b = transforms.ToTensor()(b)
        b[b>0.9] = 0
        b[b!=0] = 1
        mask = b.clone()
        b = transforms.ToPILImage()(b)
        img2 = Image.composite(layer, bkg_img, b).convert('RGB')        # |
        
        img2 = np.array(img2).astype(np.float64)                        # -><numpy>[H W C]
        img2 = img2[:, :, ::-1]                                         # Change RGB to BGR -<numpy>[H W C]
        img2 -= self.mean_bgr                                           # minus channel wise mean(BGR)
        img2 = img2.transpose((2, 0, 1))                                # -<numpy>[H W C]->[C H W]
        img2 = torch.from_numpy(img2.copy()).float()
        img2 = img2.cuda() if self.cuda else img2
        img2 = img2.unsqueeze(0) if len(img2.size()) == 3 else img2     

        mask = mask.cuda() if self.cuda else mask

        return img1, img2, mask

    
    def AddingWaterMarkAll(self, bkg_path, wm_path, start_point=(224-80,0)):
        bkg_img = Image.open(bkg_path)
        bkg_img = self.transform4VGG(bkg_img)               # <Image>Size(224,224)
        img1 = np.array(bkg_img).astype(np.float64)         # -><numpy array>[H W C]
        
        img1 = img1[:, :, ::-1]                             # Change RGB to BGR -[H W C]
        img1 -= self.mean_bgr                               # minus channel wise mean(BGR)
        img1 = img1.transpose((2, 0, 1))                    # [H W C]->[C H W]
        img1 = torch.from_numpy(img1.copy()).float()
        img1 = img1.cuda() if self.cuda else img1
        img1 = img1.unsqueeze(0) if len(img1.size()) == 3 else img1     

        # bkgPlusWM 
        wm_img = Image.open(wm_path)                                    # <Image>
        
        wm_img = self.transform4wm(wm_img)                              # Resize Watermark

        layer = Image.new('RGBA', bkg_img.size, (0, 0, 0, 0))           
        layer.paste(wm_img, start_point)                                
        r, g, b, a = layer.split()                                      
        b = transforms.ToTensor()(b)
        b[b>0.9] = 0
        b[b!=0] = 1
        b = transforms.ToPILImage()(b)
        img2 = Image.composite(layer, bkg_img, b).convert('RGB')        # |
        
        img2 = np.array(img2).astype(np.float64)                        # -><numpy>[H W C]
        img2 = img2[:, :, ::-1]                                         # Change RGB to BGR -<numpy>[H W C]
        img2 -= self.mean_bgr                                           # minus channel wise mean(BGR)
        img2 = img2.transpose((2, 0, 1))                                # -<numpy>[H W C]->[C H W]
        img2 = torch.from_numpy(img2.copy()).float()
        img2 = img2.cuda() if self.cuda else img2
        img2 = img2.unsqueeze(0) if len(img2.size()) == 3 else img2     

        # mask : generate the mask(if logo area 1，otherwise 0)
        mask = torch.zeros(bkg_img.size)
        w, h = wm_img.size

        mask[start_point[1]:start_point[1]+h,start_point[0]:start_point[0]+w] = 1
        mask = mask.cuda() if self.cuda else mask

        return img1, img2, mask

    def save4image(self, img, save_path):
        # process
        img = img.detach().cpu().numpy()
        img = img.transpose((0, 2, 3, 1))                     # BGR [N C H W] -> BGR  [N H W C]
        img = img + self.mean_bgr                             # Add channel-wise mean
        img = img[:, :, :, ::-1] / 255.                       # BGR->RGB ; [0,255]->[0,1]
        img = img.transpose((0, 3, 1, 2))                     # RGB [N C H W]
        img = torch.from_numpy(img.copy()).clamp_(0, 1)       # tensor

        save_image(img, save_path)                            # save image to 'save_path'
    
    def save4maskzone(self,img,mask,save_path):
        # process
        img = img.detach().cpu().numpy()
        img = img.transpose((0, 2, 3, 1))                     # BGR [N C H W] -> BGR  [N H W C]
        img = img + self.mean_bgr                             # Add channel-wise mean
        img = img[:, :, :, ::-1] / 255.                       # BGR->RGB ; [0,255]->[0,1]
        img = img.transpose((0, 3, 1, 2))                     # RGB [N C H W]
        img = torch.from_numpy(img.copy()).clamp_(0, 1)       # tensor
        img = img * mask.cpu()

        save_image(img, save_path)
    
    
    def save4rtot(self, r_tot, mask, save_path):
        r_tot = r_tot.cpu().numpy()                             # BGR
        r_tot = r_tot.transpose((0, 2, 3, 1))                   # BGR [N C H W] -> BGR  [N H W C]
        r_tot = r_tot + self.mean_bgr
        r_tot = r_tot[:, :, :, ::-1] / 255.                     # BGR [0,255] -> RGB [0,1]
        
        r_tot = r_tot.transpose((0, 3, 1, 2))                   # RGB [N C H W]
        r_tot = torch.from_numpy(r_tot.copy()).clamp_(0, 1) * mask.cpu()

        save_image(r_tot, save_path)

    def evalimage(self, img):
        self.net.eval()
        f = self.net(img)
        probs = self.softmax(f)

        p = probs.max(dim=1)[0].data
        pred_lbl = probs.max(dim=1)[1].data
        return p.item(), pred_lbl.item()

    def findProb(self, img, idx):
        self.net.eval()
        f = self.net(img)
        probs = self.softmax(f)

        p = probs[0][idx].item()
        return p
    
    def project_lp(self,v,xi,p):
        if p == 2:
            v = v * min(1, xi / torch.norm(v,p=2))
        elif p == np.inf:
            cmp = torch.ones_like(v).cuda() if self.cuda else torch.ones_like(v)
            cmp = xi * cmp
            v = torch.sign(v) * torch.min(abs(v),cmp)
        else:
            raise NotImplementedError

        return v

    def deepWMfool(self, image, imgPlusWM, mask, topk=20, overshoot=0.02, max_iter=5000):
        '''
        |DeepFool with mask Algorithm|
        image    : Original image
        imgPlusWM: Original image+logo
        topk     : Compute perturbations for topk classes
        overshoot: Forced perturbation change
        max_iter : The maximum number of iterations
        '''
        self.net.eval()
        if len(image.size()) == 3:
            c, h, w = image.size()
            image = image.clone().view(1, c, h, w)
            imgPlusWM = imgPlusWM.clone().view(1, c, h, w)
        image.requires_grad = True
        imgPlusWM.requires_grad = True

        f_image = self.net.forward(imgPlusWM)
        I = f_image.view(-1).sort(descending=True)[1]           
        I = I[0: topk]                                          
        label = I[0]                                            

        w = torch.zeros_like(image).cuda() if self.cuda else torch.zeros_like(image)
        r_tot = torch.zeros_like(image).cuda() if self.cuda else torch.zeros_like(image)

        loop_i = 0

        x = imgPlusWM.clone().detach()
        x.requires_grad = True
        fs = self.net(x)
        fs_list = [fs[0, I[k]] for k in range(topk)]
        k_i = label

        while k_i == label and loop_i < max_iter:     
            pert = np.inf                             
            fs[0, I[0]].backward(retain_graph=True)
            grad_orig = x.grad.data.clone()

            for k in range(1, topk):
                zero_gradients(x)

                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.clone()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data

                pert_k = abs(f_k)/torch.norm(w_k, p=2)

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i = (pert+1e-4) * w / torch.norm(w, p=2)  
            r_tot = r_tot + r_i

            pert_image = imgPlusWM + (1 + overshoot) * r_tot * mask  

            x = pert_image.clone().detach()
            x.requires_grad = True
            fs = self.net(x)
            probs = self.softmax(fs)
            k_i = torch.argmax(fs.data)

            loop_i += 1
            if loop_i % 100 == 0:
                print(loop_i, probs[0][k_i].item())

        r_tot = (1+overshoot) * r_tot * mask


        return r_tot, loop_i, label, k_i, probs[0][label].item(), probs[0][k_i].item(), pert_image


if __name__ == "__main__":
    pass
