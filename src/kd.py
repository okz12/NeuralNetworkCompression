import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss():

    def __init__(self, T, alpha):
        self.T = T
        self.alpha = alpha

    def __call__(self, y, labels, teacher_scores):
        alpha = self.alpha
        T = self.T

        # cross-entropy loss from mismatched training labels
        label_loss = F.cross_entropy(y, labels) 

        # KL-Divergence loss from teacher/student output distributions
        teacher_loss = nn.KLDivLoss()(F.log_softmax(y/T, dim=0), F.softmax(teacher_scores/T, dim=0)) * (T*T * 2.0)
        return  (label_loss * (1. - alpha)) + (teacher_loss * alpha)


class kd_ce_loss:

    def __init__(self, temperature = 5, alpha = 0.8):
        self.temperature = temperature
        self.alpha = alpha


    def __call__(self, logits_S, labels, logits_T):
        '''
        Calculate the cross entropy between logits_S and logits_T
        :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
        '''
        temperature, alpha = self.temperature, self.alpha

        if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
            temperature = temperature.unsqueeze(-1)
        beta_logits_T = logits_T / temperature
        beta_logits_S = logits_S / temperature
        p_T = F.softmax(beta_logits_T, dim=-1)
        distillation_loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
        target_loss = nn.CrossEntropyLoss()(logits_S, labels)
        loss = alpha * distillation_loss + (1 - alpha) * target_loss
        return loss


# def kd_ce_loss(logits_S, labels, logits_T, temperature=5, alpha = 0.8):
#     '''
#     Calculate the cross entropy between logits_S and logits_T
#     :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
#     :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
#     :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
#     '''
#     if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
#         temperature = temperature.unsqueeze(-1)
#     beta_logits_T = logits_T / temperature
#     beta_logits_S = logits_S / temperature
#     p_T = F.softmax(beta_logits_T, dim=-1)
#     distillation_loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
#     target_loss = nn.CrossEntropyLoss()(logits_S, labels)
#     loss = alpha * distillation_loss + (1 - alpha) * target_loss
#     return loss


# def kd_ce_loss(outputs, targets, temp=2):
#     outputs = nn.LogSoftmax(dim=1)(outputs/temp)
#     return -torch.mean(torch.sum(targets * outputs, dim=1)) * temp ** 2

def distill_unlabeled(y, teacher_scores, T=7):
    return nn.KLDivLoss()(F.log_softmax(y/T), F.softmax(teacher_scores/T)) * T * T


def extract_logits(model, trainloader):
    model.eval()
    for i, (inputs, _) in enumerate(trainloader):
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = model(inputs)
            if i == 0:
                logits = outputs.clone()
            else:
                logits = torch.cat([logits, outputs.clone()], dim=0)
    return logits