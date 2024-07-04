import torch
import torch.nn.functional as F

class VITriLoss(torch.nn.Module):
    def __init__(self, margin=0.5):
        super(VITriLoss, self).__init__()
        self.margin = margin

    def forward(self, image_features, object_ids, label_features, device):

        # select the anchor (object center of each sample)
        anchor_features = label_features[object_ids,:,:]

        random_indices = torch.randint(0, anchor_features.size(1), (anchor_features.size(0),))
        expanded_indices = random_indices.unsqueeze(-1).expand(-1, anchor_features.size(2)).to(device)
        new_anchor_features = torch.gather(anchor_features, 1, expanded_indices.unsqueeze(1)).squeeze(1)

        # compute triplet loss:
        positive_similarity = F.cosine_similarity(image_features, new_anchor_features)
        positive_distance = 1 - positive_similarity

        # losses = F.relu(positive_distance - negative_distance + self.margin)
        losses = F.relu(positive_distance + self.margin)
        count_nonzero = torch.count_nonzero(losses)

        # Average the loss over the number of valid triplets
        return torch.sum(losses) / count_nonzero if count_nonzero > 0 else torch.sum(losses)