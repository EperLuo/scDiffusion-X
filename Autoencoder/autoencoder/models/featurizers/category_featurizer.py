import torch 
import torch.nn.functional as F

class CategoricalFeaturizer(torch.nn.Module):
    def __init__(self, n_cat, one_hot_encode_features, device, embedding_dimensions=None):
        """
        Categorical feature embedding module.

        Args:
            n_cat (int): Number of categories.
            one_hot_encode_features (bool): Whether to one-hot encode features.
            device (torch.device): Device to place the module on.
            embedding_dimensions (int, optional): Number of dimensions for embeddings. Defaults to None.
        """
        super().__init__()
        self.n_cat = n_cat
        self.device = device
        self.one_hot_encode_features = one_hot_encode_features
        if not self.one_hot_encode_features:
            self.embeddings = torch.nn.Embedding(n_cat, embedding_dimensions).to(self.device)
    
    def forward(self, obs):
        """Extract features 

        Args:
            obs (torch.Tensor): The batch of observations 

        Returns:
            torch.Tensor: Extracted embeddings 
        """
        obs = obs.to(self.device)
        if self.one_hot_encode_features: 
            return F.one_hot(obs, num_classes=self.n_cat).float()
        else:
            return self.embeddings(obs)
        