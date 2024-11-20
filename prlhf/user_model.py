import numpy as np
import torch
from torch import nn


class UserModel(nn.Module):
    """
    The implicit user model f^{im}_{P} (i).
    """
    def __init__(self, tokenizer, n_user_tokens, seed):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_user_tokens = n_user_tokens
        self.seed = seed
    
    @classmethod
    def get_user_ids(cls, tokenizer, user_identifiers):
        """
        Args:
            tokenizer: a tokenizer.
            user_identifiers: a tensor or a list of tokenized user identifiers u^p.
        """
        device = user_identifiers[0].device if isinstance(user_identifiers, list) \
            else user_identifiers.device
        # decode tokenized user identifiers into text
        # the text should be in the format of "USER: {i}", where i is the user index
        user_identifier_txt = tokenizer.batch_decode(
            user_identifiers, skip_special_tokens=True)
        user_ids = [int(t.replace("USER: ", "").strip()) for t in user_identifier_txt]
        user_ids = torch.tensor(np.asarray(user_ids), device=device)
        return user_ids


class IndividualUserModel(UserModel):
    """
    Individualized Preference as in Example 2 in the paper.
    User embeddings are implemented as an embedding table.
    """
    def __init__(self, tokenizer, user_embed_dim, n_users=10, n_user_tokens=1, 
                 init_value=None, random_range=0.5, seed=123, add_generic_user=True):
        super().__init__(tokenizer, n_user_tokens, seed)
        self.set_user_embedding(n_users, n_user_tokens, user_embed_dim, 
                                init_value, random_range, seed)
        self.add_generic_user = add_generic_user
        
    def set_user_embedding(self, n_users, n_user_tokens, user_embed_dim, 
                           init_value, random_range, seed):
        self.n_users = n_users

        if init_value is not None:
            assert init_value.shape == ((n_users+1)*n_user_tokens, user_embed_dim), \
                "The shapes of init_value and user_embeddings do not align!"
        else:
            torch.manual_seed(seed)
            init_value = torch.FloatTensor((n_users+1)*n_user_tokens, user_embed_dim).uniform_(
                -random_range, random_range
            )
        
        # for unknown users, user index = 0 (generic implicit user embeddings)
        # for known users, user index = 1, 2, ...
        self.user_embedding = nn.Embedding((n_users+1)*n_user_tokens, user_embed_dim)
        self.user_embedding.weight.data = init_value
        
    def get_user_embeddings(self, user_identifiers, generic_user=False):
        """
        Args:
            user_identifiers: a tensor or a list of tokenized user identifiers.
            generic_user: if set to True, return the generic implicit user embedding e^{im}_0,
            else return the individual implicit user embeddings e^{im}_i.
        """
        device = user_identifiers[0].device if isinstance(user_identifiers, list) \
            else user_identifiers.device
        generic_user_ids = torch.tensor(np.asarray([0]*len(user_identifiers)), device=device)
        generic_user_ids = torch.concat(
            [generic_user_ids*self.n_user_tokens + i for i in range(self.n_user_tokens)])
        # (n_user_tokens x batch_size, user_embed_dim)
        user_embeddings = self.user_embedding(generic_user_ids)
        # (n_user_tokens, batch_size, user_embed_dim)
        user_embeddings = user_embeddings.reshape((self.n_user_tokens, len(user_identifiers), -1))
        # (batch_size, n_user_tokens, user_embed_dim)
        user_embeddings = user_embeddings.permute(1, 0, 2)

        if not generic_user:
            each_user_ids = self.get_user_ids(self.tokenizer, user_identifiers)
            each_user_ids_tokens = torch.concat(
                [each_user_ids*self.n_user_tokens + i for i in range(self.n_user_tokens)])
            # retrieve the 1st, 2nd, ... tokens for all users
            each_user_embeddings = self.user_embedding(each_user_ids_tokens)
            each_user_embeddings = each_user_embeddings.reshape((self.n_user_tokens, len(user_identifiers), -1))
            each_user_embeddings = each_user_embeddings.permute(1, 0, 2)

            if self.add_generic_user:  # e^{im}_i = e^{im}_0 + o_i
                user_embeddings += (each_user_ids != 0).unsqueeze(dim=1).unsqueeze(dim=2) * each_user_embeddings
            else:  # e^{im}_i = o_i
                user_embeddings = each_user_embeddings

        return user_embeddings
    
    def forward(self, user_identifiers):
        """
        user_identifiers: a tensor or a list of tokenized user identifiers.
        return: the implicit user embeddings e^{im}_i.
        """
        return self.get_user_embeddings(user_identifiers, generic_user=False)
    

class ClusterUserModel(UserModel):
    """
    Cluster-based Preference as in Example 3 in the paper.
    Cluster embeddings and user weights are implemented as embedding tables.
    """
    def __init__(self, tokenizer, user_embed_dim, n_users=10, n_clusters=5, 
                 n_user_tokens=1, init_value=None, random_range=0.5, seed=123):
        super().__init__(tokenizer, n_user_tokens, seed)
        self.n_clusters = n_clusters
        self.set_user_weight_and_cluster_embedding(
            n_users, n_clusters, n_user_tokens, user_embed_dim, 
            init_value, random_range, seed
        )
        
    def set_user_weight_and_cluster_embedding(
            self, n_users, n_clusters, n_user_tokens, user_embed_dim, 
            init_value, random_range, seed):
        if init_value is not None:
            assert init_value.shape == (n_clusters*n_user_tokens, user_embed_dim), \
                "The shapes of init_value and cluster_embeddings do not align!"
        else:
            torch.manual_seed(seed)
            init_value = torch.FloatTensor(n_clusters*n_user_tokens, user_embed_dim).uniform_(
                -random_range, random_range
            )
        
        self.cluster_embedding = nn.Embedding(n_clusters*n_user_tokens, user_embed_dim)
        self.cluster_embedding.weight.data = init_value
        # for unknown users, user index = 0 (generic implicit user embeddings)
        # for known users, user index = 1, 2, ...
        self.user_weight = nn.Embedding(n_users + 1, n_clusters)
        self.softmax = nn.Softmax(dim=1)
    
    def get_user_embeddings(self, user_identifiers, generic_user=False):
        """
        Args:
            user_identifiers: a tensor or a list of tokenized user identifiers.
            generic_user: if set to True, return the generic implicit user embedding e^{im}_0,
            else return the individual implicit user embeddings e^{im}_i.
        """
        device = user_identifiers[0].device if isinstance(user_identifiers, list) \
            else user_identifiers.device
        if generic_user:
            user_ids = torch.tensor(np.asarray([0]*len(user_identifiers)), device=device)
        else:
            user_ids = self.get_user_ids(self.tokenizer, user_identifiers)

        user_weights = self.user_weight(user_ids)  # (batch_size, n_clusters)
        user_weights = self.softmax(user_weights) 

        cluster_base_idx = self.n_user_tokens * torch.range(0, self.n_clusters-1, dtype=torch.int, device=device)
        user_embeddings = torch.stack(
            [torch.matmul(user_weights, self.cluster_embedding(cluster_base_idx + i)) 
             for i in range(self.n_user_tokens)], dim=1)  # (batch_size, n_user_tokens, user_embed_dim)

        return user_embeddings
    
    def forward(self, user_identifiers):
        """
        user_identifiers: a tensor or a list of tokenized user identifiers.
        return: the implicit user embeddings e^{im}_i.
        """
        return self.get_user_embeddings(user_identifiers, generic_user=False)