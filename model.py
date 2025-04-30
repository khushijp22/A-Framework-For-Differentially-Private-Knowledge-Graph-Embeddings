import torch # import torch module
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransEModel(nn.Module):
    """TransE Knowledge Graph Embedding Model"""
    
    def __init__(self, entity_count: int, relation_count: int, embedding_dim: int = 100, device: str = "cuda"):
        """
        Initialize the TransE model
        
        Args:
            entity_count: Number of entities in the knowledge graph
            relation_count: Number of relations in the knowledge graph
            embedding_dim: Dimension of the embedding vectors
            device: Device to run the model on (cuda or cpu)
        """
        super(TransEModel, self).__init__()
        
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Initialize embeddings with better scaling
        self.entity_embeddings = nn.Embedding(entity_count, embedding_dim).to(device)
        self.relation_embeddings = nn.Embedding(relation_count, embedding_dim).to(device)
        
        torch.nn.init.uniform_(
            self.entity_embeddings.weight.data, 
            -6/np.sqrt(embedding_dim), 
            6/np.sqrt(embedding_dim)
        )
        torch.nn.init.uniform_(
            self.relation_embeddings.weight.data, 
            -6/np.sqrt(embedding_dim), 
            6/np.sqrt(embedding_dim)
        )
        
        # Save initial embeddings for later comparison
        self.initial_entity_embeddings = self.entity_embeddings.weight.data.clone().cpu().numpy()
        self.initial_relation_embeddings = self.relation_embeddings.weight.data.clone().cpu().numpy()
        
    def l2_distance(self, head, relation, tail):
        """Compute L2 distance for TransE: ||h + r - t||_2."""
        return torch.norm(head + relation - tail, p=2, dim=1)
    
    def forward(self, triples, normalize=True):
        """
        Forward pass to compute scores for triples
        
        Args:
            triples: Tensor of shape (batch_size, 3) containing (head, relation, tail) triples
            normalize: Whether to normalize embeddings
            
        Returns:
            Tensor of scores (L2 distances) for each triple
        """
        heads = self.entity_embeddings(triples[:, 0])
        relations = self.relation_embeddings(triples[:, 1])
        tails = self.entity_embeddings(triples[:, 2])
        
        # Optional normalization
        if normalize:
            heads = F.normalize(heads, p=2, dim=1)
            relations = F.normalize(relations, p=2, dim=1)
            tails = F.normalize(tails, p=2, dim=1)
            
        return self.l2_distance(heads, relations, tails)
    
    def normalize_embeddings(self):
        """Normalize embeddings to unit length"""
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(self.entity_embeddings.weight.data, p=2, dim=1)
            self.relation_embeddings.weight.data = F.normalize(self.relation_embeddings.weight.data, p=2, dim=1)
    
    def get_parameters(self):
        """Get all parameters of the model"""
        return list(self.entity_embeddings.parameters()) + list(self.relation_embeddings.parameters())
    
    def get_state_dict(self):
        """Get state dict for saving the model"""
        return {
            'entity_embeddings': self.entity_embeddings.state_dict(),
            'relation_embeddings': self.relation_embeddings.state_dict()
        }
    
    def load_state_dict_from_dict(self, state_dict):
        """Load state dict from dictionary"""
        self.entity_embeddings.load_state_dict(state_dict['entity_embeddings'])
        self.relation_embeddings.load_state_dict(state_dict['relation_embeddings'])
