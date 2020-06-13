import os
import torch
from VGG16 import VGG_16
from DWMF import DeepWaterMarkFool
import pandas as pd
import numpy as np
import yaml
from PIL import Image
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

root = './images'

Logo = {
    'ins' : './logo/ins.png',
    'basa' : './logo/basa.png'
}

Start_point = {
    'left-up' : (0,0),
    'right-up' : (224-80,0),
    'left-down' : (0,224-80),
    'right-down' : (224-80,224-80)
}

# Selecting and adding the normal logo can also classify the correct images.
def selectIMG():
    
    file = './vgg_selected.csv'
    content = pd.read_csv(file).values
    
    vgg_net = VGG_16()
    vgg_net.load_weights()
    vgg_net = vgg_net.float()

    df = DeepWaterMarkFool(2622, vgg_net)

    for sp in Start_point:
        for logoName in Logo:
            f = open('files/{}_{}.csv'.format(sp,logoName),'w')
            f.writelines('path,class,prob\n')
            for i in range(len(content)):
                img_path = content[i][0]
                img_class = content[i][1]
                try:
                    orig, imgPwm, mask = df.AddingWaterMark(
                        bkg_path=os.path.join(root, img_path),
                        wm_path=Logo[logoName],
                        start_point=Start_point[sp]
                    )
                    pred_p, pred_lbl = df.evalimage(imgPwm)
                    if img_class == pred_lbl:
                        f.writelines("{},{},{}\n".format(img_path,pred_lbl,pred_p))
                except:
                    print('{} Failed!'.format(img_path))
            f.close()
# Use adversarial logo to attack a single image
def Restrict():

    # Attack image list(예시)
    Attack = [
        'Salli_Richardson-Whitfield/00000085.jpg',
        'Jenna_Fischer/00000033.jpg',
        'Spencer_Boldman/00000064.jpg',
        'Maria_Bonnevie/00000050.jpg',
        'Brian_Dennehy/00000080.jpg',
        'Alice_Cooper/00000072.jpg',
        'Nina_Dobrev/00000084.jpg',
        'Billy_Unger/00000084.jpg',
        'Debra_Winger/00000023.jpg',
        'Sophie_Okonedo/00000035.jpg',
        'Jacob_Latimore/00000059.jpg',
        'Tom_Skerritt/00000044.jpg',
        'Cory_Monteith/00000023.jpg',
        'Beau_Mirchoff/00000012.jpg'
    ]

    result_path = 'result2'
    vgg_net = VGG_16()
    vgg_net.load_weights()
    vgg_net = vgg_net.float()

    df = DeepWaterMarkFool(2622, vgg_net)
    log = open('log_constr.txt', 'w')
    for a_img_path in Attack:
        img_name = a_img_path.replace('/', '_')

        for logo in Logo:
            logo_path = Logo[logo]

            for sp in Start_point:
                start_point = Start_point[sp]

                # Process and get the Original image，Original image+logo，mask
                orig, imgPwm, mask = df.AddingWaterMark(
                    bkg_path=os.path.join(root, a_img_path),
                    wm_path=logo_path,
                    start_point=start_point
                )

                o_p , o_lbl = df.evalimage(orig)

                # Original image+logo‘s confidence of prediction and label of prediction
                orig_p, orig_lbl = df.evalimage(imgPwm)

                if(o_lbl != orig_lbl):
                    continue

                # save the original image+logo
                df.save4image(imgPwm, os.path.join(result_path, 'WM_{}_{}_'.format(
                    logo,sp
                )+img_name))
                
                # r_tot：perturbation | k_i：predicted label after the attack| p_lbl：the confidence about original label（after the attack） | p_ki：the confidence about predicted the k_i class(after the attack)
                r_tot, iterations, _, k_i, p_lbl, p_ki, pert_image = df.deepWMfool(
                    orig, imgPwm, mask, max_iter=5000)

                # Write the information
                wmInfo = '# wm_{}_{} # {} | original_label  [{}] of probability {} || after attacking : pred_label [{}] of probability {}, origin class {}.// Iterations {}.\n'.format(
                        logo, sp, a_img_path,  orig_lbl, orig_p, k_i, p_ki, p_lbl, iterations
                    )
                log.writelines( wmInfo )
                # Save the image with perturbation
                df.save4image(pert_image, os.path.join(result_path, 'PertWM_{}_{}_'.format(logo,sp)+img_name))
                df.save4rtot(r_tot,mask,os.path.join(result_path, 'r_tot_WM_{}_{}_'.format(logo,sp)+img_name))
        log.writelines('\n')
    log.close()

# Generate the universal adversarial logo（iterative process）
def UAP(args):
    sp = args["sp"]
    logoName = args["logoName"]
    train_ = args["train"]
    threshold = args["threshold"]
    maxIter = args["maxiter"]
    
    f = pd.read_csv("./files/{}_{}.csv".format(sp,logoName))
    content = f.values
    train = content[train_[0]:train_[1]]

    num = len(train)
    index_order = np.arange(num)

    vgg_net = VGG_16()
    vgg_net.load_weights()
    vgg_net = vgg_net.float()

    df = DeepWaterMarkFool(2622, vgg_net)
    v = torch.zeros((1,3,224,224)).cuda() if df.cuda else torch.zeros((1,3,224,224))
    
    iter = 0
    success = 0

    while success<threshold and iter < maxIter:
        np.random.shuffle(index_order)
        print("Iteration UAP ", iter)
        success = 0
        for index in index_order:
            r_path = train[index][0]
            gt = train[index][1]
            gt_p = train[index][2]

            if args["MaskAll"] == True:
                orig, imgPwm, mask = df.AddingWaterMarkAll(
                bkg_path=os.path.join(root, r_path),
                wm_path=Logo[logoName],
                start_point=Start_point[sp]
            ) 
            else:            
                orig, imgPwm, mask = df.AddingWaterMark(
                bkg_path=os.path.join(root, r_path),
                wm_path=Logo[logoName],
                start_point=Start_point[sp]
                )

            pert_img = imgPwm + mask * v
            pred_lbl_p, pred_lbl = df.evalimage(pert_img)

            # and gt_p - pred_lbl_p < 0.1
            if gt == pred_lbl:
                r_tot, _, _, k_i, _, p_k_i, _ = df.deepWMfool(
                    orig, pert_img, mask, max_iter=5000)
                if k_i != gt or gt_p - p_k_i>0.1 :
                    v = v + r_tot
                    v = df.project_lp(v,xi=60,p=np.inf)
            else:
                print("Attack {} susscefully!".format(r_path))
                success += 1
                
            # Save the logo with perturbation and universal perturbation（sectional type）
            if success % 100 == 0:
                df.save4rtot(v,mask,args["v_path2"] + "/pert_success_{}_{}.png".format(success,iter))
                df.save4maskzone(pert_img,mask,args["maskzone_path2"] + "/logopert_success_{}_{}.png".format(success,iter))
        iter+=1
        print("Iter {} | sussess {}.".format(iter,success))

        v = v * mask
    print("Saving pickle to {}...".format(args["rtot_pkl"]))
    torch.save(v.cpu(), args["rtot_pkl"])
    df.save4rtot(v,mask,args["v_path"])
    df.save4maskzone(pert_img,mask,args["maskzone_path"])  

# In the previous process, we saved the universal adversarial logo perturbation in pkl format. Then add the perturbation in pkl format to original image+logo（correct classification） and test.
def test1(args):
    v = torch.load("./UAPResult/ins/rtot_left-down.pkl")
    v = torch.load(args["rtot_pkl"])
    v = v.cuda()
    
    sp = args["sp"]
    logoName = args["logoName"]
    test_ = args["test"]

    vgg_net = VGG_16()
    vgg_net.load_weights()
    vgg_net = vgg_net.float()

    df = DeepWaterMarkFool(2622, vgg_net)

    f = pd.read_csv("./files/{}_{}.csv".format(sp,logoName))
    content = f.values
    test = content[test_[0]:test_[1]]
    log = open(args["logfile2"],'w')
    
    success = 0
    f20 = 0
    f40 = 0
    f60 = 0
    f80 = 0
    f100 = 0
    c20 = 0
    c40 = 0
    c60 = 0
    c80 = 0
    c100 = 0
    for item in test:
        path = item[0]
        lbl = item[1]
        p = item[2]

        if args["MaskAll"] == True:
            orig, imgPwm, mask = df.AddingWaterMarkAll(
                bkg_path=os.path.join(root, item[0]),
                wm_path=Logo[logoName],
                start_point=Start_point[sp]
            ) 
        else:           
            orig, imgPwm, mask = df.AddingWaterMark(
                bkg_path=os.path.join(root, item[0]),
                wm_path=Logo[logoName],
                start_point=Start_point[sp]
            )

        pert = imgPwm + v
        pert_p, pert_lbl = df.evalimage(pert)
        ##################################
        orig_lbl_p  = df.findProb(pert,lbl)
        
        if pert_lbl != lbl:
            success +=1
            df.save4image(pert,os.path.join('./paper_result/ins/7000', 'Pertlog_'+ path.replace('/','_')))
        elif p - orig_lbl_p <= 0.2:
            f20+=1
        elif 0.2 < p - orig_lbl_p <= 0.4:
            f40+=1
        elif 0.4 < p - orig_lbl_p <= 0.6:
            f60+=1
        elif 0.6 < p - orig_lbl_p <= 0.8:
            f80+=1
        elif 0.8 < p - orig_lbl_p <= 1:
            f100+=1 
        
        if p - orig_lbl_p <= 0.2:
            c20+=1
        elif 0.2 < p - orig_lbl_p <= 0.4:
            c40+=1
        elif 0.4 < p - orig_lbl_p <= 0.6:
            c60+=1
        elif 0.6 < p - orig_lbl_p <= 0.8:
            c80+=1
        elif 0.8 < p - orig_lbl_p <= 1:
            c100+=1
        #########################################
        logInfo = '{} | orig_lbl {} orig_p {}  ///// pert_lbl {} pert_p {} original label p {}.\n'.format(
            path, lbl, p, pert_lbl,pert_p,orig_lbl_p
        )
        log.writelines(logInfo)
    log.writelines('Success: {} | f20: {} | f40: {} | f60: {} | f80: {} | f100: {} | c20: {} | c40: {} | c60: {} | c80: {} | c100: {}.'.format(success,f20,f40,f60,f80,f100,c20,c40,c60,c80,c100))
    log.close()
    print(success)

# In the previous process, we also saved the universal adversarial logo perturbation in png format. Then add the perturbation in png format to original image+logo（correct classification） and test.
def test2(args):
    sp = args["sp"]
    logoName = args["logoName"]
    test_ = args["test"]
    
    f = pd.read_csv("./files/{}_{}.csv".format(sp,logoName))
    content = f.values
    test = content[test_[0]:test_[1]]

    rtot = Image.open('./UAPResult/ins/pert_left-down_80size(1-1000-60).png')
    v = np.array(rtot).astype(np.float64)         # -><numpy array>[H W C]
        
    v = v[:, :, ::-1]                             # Change RGB to BGR -[H W C]
    v -= np.array([129.1863, 104.7624, 93.5940])     # minus channel wise mean(BGR)
    v = v.transpose((2, 0, 1))                    # [H W C]->[C H W]
    v = torch.from_numpy(v.copy()).float()
    v = v.cuda() 
    v = v.unsqueeze(0) if len(v.size()) == 3 else v     #

    vgg_net = VGG_16()
    vgg_net.load_weights()
    vgg_net = vgg_net.float()

    df = DeepWaterMarkFool(2622, vgg_net)
    log = open(args["logfile3"],'w')
    
    success = 0
    semisuccess = 0  
    for item in test:
        path = item[0]
        lbl = item[1]
        p = item[2]

        if args["MaskAll"] == True:
            orig, imgPwm, mask = df.AddingWaterMarkAll(
                bkg_path=os.path.join(root, item[0]),
                wm_path=Logo[logoName],
                start_point=Start_point[sp]
            ) 
        else:           
            orig, imgPwm, mask = df.AddingWaterMark(
                bkg_path=os.path.join(root, item[0]),
                wm_path=Logo[logoName],
                start_point=Start_point[sp]
            )

        pert = imgPwm + v * mask
        pert_p, pert_lbl = df.evalimage(pert)
        #################
        orig_lbl_p  = df.findProb(pert,lbl)
        
        if  pert_lbl != lbl:
            success+=1
            ################
            df.save4image(pert, os.path.join('./paper_result/ins/i', 'Pertlog_'+ path.replace('/','_')))
        elif p - pert_p > 0.2:
            semisuccess+=1
        
        ####################################
        logInfo = '{} | orig_lbl {} orig_p {}  ///// pert_lbl {} pert_p {} original label p {}.\n'.format(
            path, lbl, p, pert_lbl,pert_p,orig_lbl_p
        )
        log.writelines(logInfo)
    log.writelines('Success: {} | not success but decrease the confidence more than 20%: {}.'.format(success,semisuccess))
    log.close()
    
    print(success)
    print(semisuccess)





import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config',type=int,default=1)
args = parser.parse_args()


if __name__ == "__main__":
    f = open('./config.yaml')
    f = f.read()
    config = yaml.load(f)
    print(config[args.config])
    # UAP(config[args.config]) # Then generate the universal adversarial logo.
    # test1(config[args.config]) # test1(perturbation in pkl format)
    # test2(config[args.config]) # test2(perturbation in png format)
    selectIMG() # At first, select the images.