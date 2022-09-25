import torch
from torch import nn
import torch.nn.functional as F
from utils.options import args_parser

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.args = args_parser()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        self.args.device = torch.device(
            'cuda:{}'.format(self.args.gpu) if torch.cuda.is_available() and self.args.gpu != -1 else 'cpu')
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        self.temperature = self.temperature.to(self.args.device)
        self.negatives_mask = self.negatives_mask.to(self.args.device)
        similarity_matrix = similarity_matrix.to(self.args.device)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        #print(nominator)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        #print(torch.sum(denominator, dim=1))

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
