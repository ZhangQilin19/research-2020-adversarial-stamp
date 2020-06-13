import os
import torch
from VGG16 import VGG_16
from VGGFaceDataSet import VggFaceDataset
from torch.utils.data import Dataset
from torchvision.utils import save_image



########################################
# data cleaning
# -- Select the image that can be classified correctly
# -- Image path，label，class name，confidence
#
########################################

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # loading vgg-face network
    vgg_net = VGG_16()
    vgg_net.load_weights()
    vgg_net = vgg_net.float()
    vgg_net = vgg_net.cuda()

    # loading vgg-face dataset
    dataset = VggFaceDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True)

    # loss function
    criterion = torch.nn.CrossEntropyLoss()
    # softmax (calculate the confidence)
    softmax = torch.nn.Softmax(dim=1)
    
    correct = 0
    total = 0

    vgg_net.eval()
    f = open('vgg_selected.csv', 'w')
    f.write('path,class_idx,class_name,prob\n')
    for idx, batch in enumerate(dataloader):
        imgs = batch['img'].cuda()
        lbls = batch['cls_idx'].cuda()
        outputs = vgg_net(imgs)

        # Calculate the confidence of each class
        probs = softmax(outputs)

        # Confidence of predictive classification
        max_prob = probs.max(dim=1)[0]

        # Calculate the current loss(unnecessary)
        loss = criterion(outputs, lbls)

        # pred : Predicted class ; lbls ： Actual label
        # Choose the correct predicted picture to attack
        _, pred = torch.max(outputs.data, 1)
        selected = (pred == lbls)

        # recorded to'vgg_selected.csv'
        for i, s in enumerate(selected):
            if s:
                f.write("{},{},{},{}\n".format(batch['path'][i], batch['cls_idx'][i].item(),
                                               batch['cls_name'][i], max_prob[i].item()))

        total += lbls.size()[0]
        correct += selected.sum().item()

        if (idx+1) % 100 == 0:
            print("==>Processing batch-{} | loss : {}".format((idx+1), loss.item()))

    print("Acc is {}.".format(correct/total))
    print("Finished!")
