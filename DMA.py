import torch
import torch.nn as nn
import torch.nn.functional as F


def dma_loss(cosine):
    '''
    :param cosine: the cosine between w and x of the classification layer, batchsize * class_num
    :return: dma loss
    '''
    # pick the maximum cosine of each sample
    maximum_cosine = cosine.max(dim=1)[0]  # B

    # get the minimum theta of each sample
    minimum_theta = torch.acos(maximum_cosine)  # B

    # compute dma loss
    loss = 0.5 * minimum_theta.pow(2).mean()

    return loss


class DMA_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(DMA_Linear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        '''
        :param x: the input, batchsize*dimension
        :return: in addition to the normal output of linear layer, DMA_Linear computes and outputs the cosine of this layer
        '''

        # costheta
        cosine = torch.mm(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1).t().detach())  # B x class_num#
        cosine = cosine.clamp(-1, 1)  # for numerical stability , B x class_num

        return F.linear(x, self.weight, self.bias), cosine




