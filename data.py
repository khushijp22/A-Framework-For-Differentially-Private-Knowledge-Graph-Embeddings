import os
import random
import torch
from typing import List
from pykeen.datasets import PathDataset
from typing import List, Tuple, Dict, Set, Optional

class KGDataHandler:
    """Knowledge Graph Data Handler"""
    
    def __init__(self, fb_path: str, confidential_ratio: float = 0.3):
        """
        Initialize the KG data handler
        
        Args:
            fb_path: Path to the dataset files
            confidential_ratio: Fraction of training data that requires privacy protection
        """
        self.fb_path = fb_path
        self.confidential_ratio = confidential_ratio
        
        self.train_data = self.load_triplets(os.path.join(fb_path, 'train.txt'))
        self.valid_data = self.load_triplets(os.path.join(fb_path, 'valid.txt'))
        self.test_data = self.load_triplets(os.path.join(fb_path, 'test.txt'))
        
        self.train_path = os.path.join(fb_path, 'train.txt')
        self.valid_path = os.path.join(fb_path, 'valid.txt')
        self.test_path = os.path.join(fb_path, 'test.txt')
        
        self.dataset = PathDataset(
            training_path=self.train_path,
            testing_path=self.test_path,
            validation_path=self.valid_path
        )
        
        self._prepare_data()
        
    def load_triplets(self, file_path: str) -> List[List[str]]:
        """
        Load triplets from file
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of triplets [head, relation, tail]
        """
        with open(file_path, 'r') as f:
            triplets = [line.strip().split('\t') for line in f]
        return triplets
    
    def _prepare_data(self):
        """Prepare data for training"""
        # Create entity and relation dictionaries
        self.entities = set()
        self.relations = set()
        self.entities_t = set()
        self.relations_t = set()
        
        for h, r, t in self.train_data:
            self.entities.add(h)
            self.entities.add(t)
            self.relations.add(r)
            
        for h, r, t in self.test_data:
            self.entities_t.add(h)
            self.entities_t.add(t)
            self.relations_t.add(r)
            
        # Get mappings from dataset
        self.entity_to_id = self.dataset.training.entity_to_id  
        self.relation_to_id = self.dataset.training.relation_to_id
        self.entity_to_id_t = self.dataset.testing.entity_to_id
        self.relation_to_id_t = self.dataset.testing.relation_to_id
        
        self.entity_count = len(self.entity_to_id)
        self.relation_count = len(self.relation_to_id)
        
        # Convert string triples to ID triples
        self.train_triples = [
            (self.entity_to_id[h], self.relation_to_id[r], self.entity_to_id[t])
            for h, r, t in self.train_data
        ]
        
        self.test_triples = [
            (self.entity_to_id_t[h], self.relation_to_id_t[r], self.entity_to_id_t[t])
            for h, r, t in self.test_data
            if (h in self.entity_to_id and r in self.relation_to_id and t in self.entity_to_id)
        ]
        
        # Split training data into confidential and unrestricted
        split_point_conf = int(len(self.train_triples) * self.confidential_ratio)
        self.confidential_triples = self.train_triples[:split_point_conf]
        self.unrestricted_triples = self.train_triples[split_point_conf:]
        
        # All triples for filtered evaluation
        self.all_triples = self.train_triples + self.test_triples
        
    def print_data_stats(self):
        """Print statistics about the data"""
        print("--------------------------------------------------------------------------------")
        print(f"Total entities: {self.entity_count}")
        print("--------------------------------------------------------------------------------")
        print(f"Total relations: {self.relation_count}")
        print("--------------------------------------------------------------------------------")
        print(f"Confidential triples: {len(self.confidential_triples)}")
        print("--------------------------------------------------------------------------------")
        print(f"Unrestricted triples: {len(self.unrestricted_triples)}")
        print("--------------------------------------------------------------------------------")
        print(f"Testing triples: {len(self.test_triples)}")
    
    def get_positive_and_negative_samples(self, triples, batch_size, neg_ratio=10, entity_count=None):
        """
        Get positive samples and corresponding negative samples
        
        Args:
            triples: List of triples [(head, relation, tail), ...]
            batch_size: Number of positive samples
            neg_ratio: Number of negative samples per positive sample
            entity_count: Number of entities in KG
            
        Returns:
            Positive and negative samples as tensors
        """
        if entity_count is None:
            entity_count = self.entity_count
            
        if len(triples) > batch_size:
            batch_indices = random.sample(range(len(triples)), batch_size)
            batch_triples = [triples[i] for i in batch_indices]
        else:
            batch_triples = triples
            
        # Create positive samples tensor
        pos_samples = torch.tensor(batch_triples, dtype=torch.long).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create negative samples by corrupting either head or tail
        neg_samples = []
        for h, r, t in batch_triples:
            for _ in range(neg_ratio):
                if random.random() < 0.5:  # Corrupt head
                    h_corrupt = random.randint(0, entity_count - 1)
                    while h_corrupt == h:
                        h_corrupt = random.randint(0, entity_count - 1)
                    neg_samples.append((h_corrupt, r, t))
                else:  # Corrupt tail
                    t_corrupt = random.randint(0, entity_count - 1)
                    while t_corrupt == t:
                        t_corrupt = random.randint(0, entity_count - 1)
                    neg_samples.append((h, r, t_corrupt))
                    
        neg_samples = torch.tensor(neg_samples, dtype=torch.long).to("cuda" if torch.cuda.is_available() else "cpu")
        return pos_samples, neg_samples
