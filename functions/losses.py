import torch
import torch.nn as nn
import torch.nn.functional as F

# Modified SupCon loss - adding regularization, penalizing close negative examples
class Modified_NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, device="cpu", lambda_ortho=0.05):
        super(Modified_NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.lambda_ortho = lambda_ortho

    def forward(self, z, labels):
        """
        Compute the Modified NT-Xent loss for a batch of embeddings `z` using the true labels.
        :param z: Tensor of shape (batch_size, latent_dim) - Latent representations.
        :param labels: Tensor of shape (batch_size,) - True labels of the samples.
        :return: NT-Xent loss scalar.
        """
        z = z.to(self.device)
        labels = labels.to(self.device)

        # Normalize embeddings
        z = F.normalize(z, dim=1, eps=1e-8)

        # Compute cosine similarity
        similarity_matrix = torch.matmul(z, z.T)  # (batch_size, batch_size)

        # Temperature scaling
        logits = similarity_matrix / self.temperature

        # Create mask for positive pairs (same label)
        labels_expand = labels.unsqueeze(0)  # Shape: (1, batch_size)
        positive_mask = torch.eq(labels_expand, labels_expand.T).to(self.device)  # (batch_size, batch_size)

        # Self-similarity removal
        mask = torch.eye(z.shape[0], dtype=torch.bool).to(self.device)
        logits.masked_fill_(mask, -1e10)  # Ignore self-pairs

        # Convert mask to probability distribution over positives
        exp_logits = torch.exp(logits)

        positives = exp_logits * positive_mask.float() + 1e-8
        negatives = exp_logits * (~positive_mask).float() + 1e-8

        # Compute loss: sum over positive pairs / sum over all pairs
        positive_sum = torch.sum(positives, dim=1)
        negative_sum = torch.sum(negatives, dim=1)

        # contrastive_loss = -torch.log(positive_sum / (positive_sum + negative_sum + 1e-8))
        contrastive_loss = -torch.log(torch.clamp(positive_sum, min=1e-8) /
                              torch.clamp(positive_sum + negative_sum, min=1e-8))


        # Orthogonality regularization: penalize similarity between negative pairs
        # Here we compute the mean similarity between all negative pairs (those with different labels)
        negative_similarity = similarity_matrix * (~positive_mask).float()
        ortho_penalty = torch.sum(negative_similarity ** 2) / (z.shape[0] * (z.shape[0] - 1))

        # Total loss: contrastive loss + orthogonality regularization
        total_loss = contrastive_loss.mean() + self.lambda_ortho * ortho_penalty

        return total_loss

