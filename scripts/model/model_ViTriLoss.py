import torch
import torch.nn.functional as F

class VITriLoss(torch.nn.Module):
    def __init__(self, margin=0.5):
        super(VITriLoss, self).__init__()
        self.margin = margin

    def forward(self, image_features, object_ids, center_features):
        
        # find negtive sample (nearest feature center)
        cosine_sim_matrix = torch.mm(image_features, center_features.T)
        mask = torch.arange(center_features.size(0), device=center_features.device).unsqueeze(0) == object_ids.unsqueeze(1)
        cosine_sim_matrix[mask] = float('-inf')
        min_sim_values, negative_indices = cosine_sim_matrix.max(dim=1)
        negative_features = center_features[negative_indices]

        # select the anchor (object center of each sample)
        anchor_features = center_features[object_ids]

        # compute triplet loss:
        positive_similarity = F.cosine_similarity(image_features, anchor_features)
        negative_similarity = F.cosine_similarity(image_features, negative_features)

        positive_distance = 1 - positive_similarity
        negative_distance = 1 - negative_similarity

        # losses = F.relu(positive_distance - negative_distance + self.margin)
        losses = F.relu(positive_distance + self.margin)
        count_nonzero = torch.count_nonzero(losses)

        # Average the loss over the number of valid triplets
        return torch.sum(losses) / count_nonzero if count_nonzero > 0 else torch.sum(losses)