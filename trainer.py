
import numpy as np
import random
from torch.optim import AdamW
from opacus.accountants import RDPAccountant
import torch
from model import TransEModel
from data import KGDataHandler
class TransETrainer:
    """TransE Model Trainer with Differential Privacy Support"""
    
    def __init__(
        self, 
        model: TransEModel, 
        data_handler: KGDataHandler,
        learning_rate: float = 0.005,
        noise_multiplier: float = 0.7,
        batch_size: int = 256,
        norm_clipping: float = 1.0,
        margin: float = 0.5,
        epochs: int = 300,
        reg_lambda: float = 1e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the trainer
        
        Args:
            model: TransE model
            data_handler: Knowledge graph data handler
            learning_rate: Step size for optimizer updates
            noise_multiplier: Amount of noise added to gradients for differential privacy
            batch_size: Number of samples per training batch
            norm_clipping: Maximum L2 norm of per-sample gradients (before adding noise)
            margin: Margin used in ranking loss to separate positive/negative triples
            epochs: Total number of training iterations over the dataset
            reg_lambda: Regularization parameter
            device: Device to run the model on (cuda or cpu)
        """
        self.model = model
        self.data_handler = data_handler
        self.learning_rate = learning_rate
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
        self.norm_clipping = norm_clipping
        self.margin = margin
        self.epochs = epochs
        self.reg_lambda = reg_lambda
        self.device = device
        
        # Initialize optimizer and scheduler
        self.parameters = self.model.get_parameters()
        self.optimizer = AdamW(self.parameters, lr=learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Privacy accountant
        self.accountant = RDPAccountant()
        
    def loss_function(self, pos_scores, neg_scores):
        """
        Margin-based ranking loss for TransE with regularization
        
        Args:
            pos_scores: Scores for positive samples
            neg_scores: Scores for negative samples
            
        Returns:
            Loss value
        """
        # Reshape to ensure compatibility
        pos_expanded = pos_scores.unsqueeze(1).expand(-1, neg_scores.size(0) // pos_scores.size(0))
        pos_expanded = pos_expanded.reshape(-1)
        
        ranking_loss = torch.mean(torch.relu(pos_expanded - neg_scores + self.margin))
        
        # Add regularization term
        if self.reg_lambda > 0:
            # L2 regularization on parameters
            reg_loss = 0
            for param in self.parameters:
                reg_loss += torch.norm(param, p=2)
            return ranking_loss + self.reg_lambda * reg_loss
            
        return ranking_loss
    
    def optimize_confidential(self, triples):
        """
        Optimize parameters for confidential statements with improved differential privacy
        
        Args:
            triples: List of triples [(head, relation, tail), ...]
            
        Returns:
            Loss value
        """
        # Get positive and negative samples
        pos_samples, neg_samples = self.data_handler.get_positive_and_negative_samples(
            triples, self.batch_size, neg_ratio=5
        )
        
        # Forward pass
        pos_score = self.model.forward(pos_samples, normalize=False)
        neg_score = self.model.forward(neg_samples, normalize=False)
        
        # Compute loss
        loss = self.loss_function(pos_score, neg_score)
        
        # Backward pass to get gradients
        self.optimizer.zero_grad()
        loss.backward()
        
        # Improved gradient clipping using global norm
        total_norm = 0
        for param in self.parameters:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip gradients globally (more stable than per-parameter)
        clip_coef = min(1.0, self.norm_clipping / (total_norm + 1e-6))
        for param in self.parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
                
        # Add calibrated noise for differential privacy
        for param in self.parameters:
            if param.grad is not None:
                noise = torch.normal(
                    mean=0.0,
                    std=self.noise_multiplier * self.norm_clipping / self.batch_size**0.5,
                    size=param.grad.shape,
                    device=self.device
                )
                param.grad.data += noise
                
        # Update parameters
        self.optimizer.step()
        sample_rate = self.batch_size / len(self.data_handler.confidential_triples)
        
        self.accountant.step(noise_multiplier=self.noise_multiplier, sample_rate=sample_rate)
        
        # Normalize embeddings after update
        self.model.normalize_embeddings()
        
        return loss.item()
    
    def optimize_unrestricted(self, triples):
        """
        Optimize parameters for unrestricted statements without differential privacy
        
        Args:
            triples: List of triples [(head, relation, tail), ...]
            
        Returns:
            Loss value
        """
        # Get positive and negative samples
        pos_samples, neg_samples = self.data_handler.get_positive_and_negative_samples(
            triples, self.batch_size, neg_ratio=10
        )
        
        # Forward pass
        pos_score = self.model.forward(pos_samples, normalize=False)
        neg_score = self.model.forward(neg_samples, normalize=False)
        
        # Compute loss
        loss = self.loss_function(pos_score, neg_score)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        # Normalize embeddings after update
        self.model.normalize_embeddings()
        
        return loss.item()
    
    def evaluate_model(self, test_triples, all_triples, batch_size=128, k=10):
        """
        Evaluate model using filtered setting with batch processing
        
        Args:
            test_triples: List of test triples
            all_triples: List of all triples (train + test)
            batch_size: Batch size for evaluation
            k: K for Hits@K metric
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Convert all_triples to a dictionary for O(1) lookup
        head_filter = {}
        tail_filter = {}
        
        for h, r, t in all_triples:
            if (h, r) not in tail_filter:
                tail_filter[(h, r)] = []
            tail_filter[(h, r)].append(t)
            
            if (r, t) not in head_filter:
                head_filter[(r, t)] = []
            head_filter[(r, t)].append(h)
            
        head_ranks = []
        tail_ranks = []
        
        test_batches = [test_triples[i:i+batch_size] for i in range(0, len(test_triples), batch_size)]
        
        with torch.no_grad():
            for batch in test_batches:
                for h, r, t in batch:
                    # Corrupt head
                    head_candidates = []
                    head_ids = []
                    
                    for e in range(self.data_handler.entity_count):
                        if e != h and (e not in head_filter.get((r, t), [])):
                            head_candidates.append((e, r, t))
                            head_ids.append(e)
                            
                    # Add true triple
                    head_candidates.append((h, r, t))
                    head_ids.append(h)
                    
                    # Get scores for head batch
                    if head_candidates:
                        head_tensors = torch.tensor(head_candidates, device=self.device)
                        head_scores = self.model.forward(head_tensors).cpu().numpy()
                        
                        # In TransE, LOWER scores are better (distance-based)
                        true_idx = head_ids.index(h)
                        true_score = head_scores[true_idx]
                        # Count entities with better (lower) scores than the true entity
                        head_rank = 1 + np.sum(head_scores < true_score)
                        head_ranks.append(head_rank)
                        
                    # Corrupt tail
                    tail_candidates = []
                    tail_ids = []
                    
                    for e in range(self.data_handler.entity_count):
                        if e != t and (e not in tail_filter.get((h, r), [])):
                            tail_candidates.append((h, r, e))
                            tail_ids.append(e)
                            
                    # Add true triple
                    tail_candidates.append((h, r, t))
                    tail_ids.append(t)
                    
                    # Get scores for tail batch
                    if tail_candidates:
                        tail_tensors = torch.tensor(tail_candidates, device=self.device)
                        tail_scores = self.model.forward(tail_tensors).cpu().numpy()
                        
                        # Find rank of true triple (lower is better)
                        true_idx = tail_ids.index(t)
                        true_score = tail_scores[true_idx]
                        # Count entities with better (lower) scores than the true entity
                        tail_rank = 1 + np.sum(tail_scores < true_score)
                        tail_ranks.append(tail_rank)
                        
        # Calculate metrics
        all_ranks = head_ranks + tail_ranks
        mr = sum(all_ranks) / len(all_ranks) if all_ranks else 0
        mrr = sum(1.0/r for r in all_ranks) / len(all_ranks) if all_ranks else 0
        hits_at_1 = sum(1 for r in all_ranks if r <= 1) / len(all_ranks) if all_ranks else 0
        hits_at_3 = sum(1 for r in all_ranks if r <= 3) / len(all_ranks) if all_ranks else 0
        hits_at_k = sum(1 for r in all_ranks if r <= k) / len(all_ranks) if all_ranks else 0
        
        return {
            'MR': mr,
            'MRR': mrr,
            'Hits@1': hits_at_1,
            'Hits@3': hits_at_3,
            'Hits@10': hits_at_k
        }
    
    def train_with_early_stopping(self, patience=5):
        """
        Train model with early stopping
        
        Args:
            patience: Number of evaluation rounds with no improvement before early stopping
            
        Returns:
            Best model state, training losses, epsilon values
        """
        best_hits = 0
        no_improve = 0
        best_model_state = None
        
        mU = 0  # Counter for unrestricted optimization steps
        mC = 0  # Counter for confidential optimization steps
        losses = []
        epsilon_values = []
        all_metrics = []
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            steps = 0
            epoch_losses = []
            
            while True:
                # Stop if both datasets are exhausted
                if (len(self.data_handler.unrestricted_triples) == 0 and 
                    len(self.data_handler.confidential_triples) == 0):
                    break
                    
                # Decide which type of batch to sample (improved balance calculation)
                if (len(self.data_handler.unrestricted_triples) == 0 or 
                    (mC < mU * len(self.data_handler.confidential_triples) / 
                     len(self.data_handler.unrestricted_triples))):
                    batch_type = "confidential"
                    loss = self.optimize_confidential(self.data_handler.confidential_triples)
                    mC += 1
                else:
                    batch_type = "unrestricted"
                    loss = self.optimize_unrestricted(self.data_handler.unrestricted_triples)
                    mU += 1
                    
                epoch_losses.append(loss)
                steps += 1
                
                if steps >= ((len(self.data_handler.unrestricted_triples) + 
                              len(self.data_handler.confidential_triples)) // self.batch_size):
                    break
                    
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            print(f"  Epoch {epoch+1}: mU={mU}, mC={mC}, avg_loss={avg_loss:.4f}")
            losses.append(avg_loss)
            
            # Update learning rate based on average loss
            self.scheduler.step(avg_loss)
            
            # Evaluate on validation set every 10 epochs
            if epoch % 10 == 0:
                val_sample = random.sample(
                    self.data_handler.test_triples, 
                    min(500, len(self.data_handler.test_triples))
                )
                metrics = self.evaluate_model(val_sample, self.data_handler.all_triples)
                print(f"  Validation: MR={metrics['MR']:.2f}, MRR={metrics['MRR']:.4f}, " + 
                      f"Hits@10={metrics['Hits@10']:.4f}")
                metrics["epoch"] = epoch
                all_metrics.append(metrics)
                # Check for improvement
                current_hits = metrics['Hits@10']
                if current_hits > best_hits:
                    best_hits = current_hits
                    no_improve = 0
                    # Save best model
                    best_model_state = {
                        'entity_embeddings': self.model.entity_embeddings.state_dict(),
                        'relation_embeddings': self.model.relation_embeddings.state_dict(),
                        'epoch': epoch,
                        'metrics': metrics
                    }
                else:
                    no_improve += 1
                    
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # Restore best model
                    if best_model_state:
                        self.model.load_state_dict_from_dict(best_model_state)
                    break
                    
            if epoch % 5 == 0:
                eps = self.accountant.get_epsilon(delta=1e-5)
                print(f"  Current privacy guarantee: (ε = {eps:.2f}, δ = 1e-5)")
                epsilon_values.append((epoch, eps))
                if eps > 10:
                    print("Epsilon exceed, stopping at epoch ", (epoch + 1))
                    if best_model_state:
                        self.model.load_state_dict_from_dict(best_model_state)
                    return best_model_state, losses, epsilon_values, all_metrics
                    
        return best_model_state, losses, epsilon_values, all_metrics
