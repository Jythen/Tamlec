import random
import torch
from torch.utils.data import Sampler
import numpy as np
import copy
from tqdm import tqdm


class SubtreeSampler(Sampler):
    """SubtreeSampler Class

    This class is a custom data sampler designed for tasks involving hierarchical taxonomies. It manages the sampling of subtrees and implements various collation strategies tailored to different algorithms.

    Attributes:
        cfg (dict): Configuration dictionary
        dataset (list): List of subtrees, where each subtree is represented as a tuple with:
            - torch.tensor: tokenized documents.
            - torch.tensor: labels.
            - torch.tensor: Column indices of relevant labels.
        batch_size (int): Number of examples in a batch.
        possible_subtrees (list): List of subtree indices.
        items_idx_per_label (dict): Mapping of subtree indices to dictionaries that map labels to sets of item indices.

    Methods:
        __len__():
            Returns the number of subtrees in the dataset.

        __iter__():
            Iterates through the subtrees, yielding subtree indices.

        collate_standard(seed=None):
            Generates a closure for standard collation, creating mini-batches of inputs and labels.

        collate_tamlec(seed=None):
            Generates a closure for Tamlec collation, creating mini-batches while maintaining taxonomy-specific constraints.
    """

    def __init__(self, dataset, cfg, batch_size):
        super().__init__(data_source=None)
        self.cfg = cfg
        self.taxonomy = self.cfg['taxonomy']
        self.dataset = dataset
        self.batch_size = batch_size
        self.possible_subtrees = list(range(len(self.dataset)))


    def __len__(self):
        return len(self.possible_subtrees)


    def __iter__(self):
        for subtree_idx in self.possible_subtrees:
            self.subtree_idx = subtree_idx
            self.sampled_labels = self.dataset[subtree_idx][2].tolist()

            yield subtree_idx


    def collate_standard(self, seed=None):
        def _collate_standard(input_data):
            """Collate function for standard classification tasks. The seed can be defined in the closure so that in training we have random batches while in evaluation we have always the same batches.

            Args:
                input_data (list): A list of one tuple containing three elements:
                    - A torch.tensor containing the inputs for the model.
                    - A torch.tensor containing the labels.
                    - A torch.tensor with relevant column indices, i.e. labels that appear in this subtree.

            Returns:
                tuple: A tuple containing three elements:
                    - A list of torch.tensor containing the inputs for the model.
                    - A list of torch.tensor containing the labels.
                    - A torch.tensor with relevant column indices, i.e. labels that appear in this subtree.
            """

            document_data = input_data[0][0]
            labels_data = input_data[0][1]
            column_indices = input_data[0][2]

            # Create mini-batches
            indices = np.arange(len(document_data))
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(indices)
            n_batches = int(np.ceil(len(indices)/self.batch_size))
            batches_indices = [indices[i*self.batch_size:(i+1)*self.batch_size] for i in range(0, n_batches)]
            # Remove empty batch if any
            batches_indices = [batch for batch in batches_indices if len(batch) != 0]
            # Create batched input and labels
            batched_input = [document_data[batch] for batch in batches_indices]
            batched_labels = [labels_data[batch] for batch in batches_indices]

            return (
                batched_input,
                batched_labels,
                column_indices,
            )
        return _collate_standard


    def collate_tamlec(self, seed=None):
        def _collate_tamlec(input_data):
            """Collate function for  Tamlec. The seed can be defined in the closure so that in training we have random batches while in evaluation we have always the same batches.

            Args:
                input_data (list): A list of one tuple containing four elements:
                    - A torch.tensor containing the inputs for the model.
                    - A list of lists containing the labels.
                    - A torch.tensor with relevant column indices, i.e. labels that appear in this subtree.
                    - An integer representing the task id

            Returns:
                tuple: A tuple containing three elements:
                    - A list of torch.tensor containing the inputs for the model.
                    - A list of lists of lists containing the labels.
                    - A list with relevant column indices, i.e. labels that appear in this subtree.
            """

            document_data = input_data[0][0]
            labels_data = input_data[0][1]
            column_indices = input_data[0][2]
            task_id = input_data[0][3]

            # Create mini-batches
            indices = np.arange(len(document_data))
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(indices)
            n_batches = int(np.ceil(len(indices)/self.batch_size))
            batches_indices = [indices[i*self.batch_size:(i+1)*self.batch_size] for i in range(0, n_batches)]
            # Remove empty batch if any
            batches_indices = [batch for batch in batches_indices if len(batch) != 0]
            # Create batched input and labels
            batched_input = []
            batched_labels = []
            if self.cfg['method'] == 'tamlec':
                for batch in batches_indices:
                    # Use slicing since document_data is a tensor
                    batched_input.append(document_data[batch])
                    # Labels: keep all labels appearing in the sub-tree (root included since tamlec requires the start of the path)
                    labels_batch = []
                    for idx in batch:
                        labels_batch.append([self.cfg['task_to_subroot'][task_id]] + [lab for lab in labels_data[idx] if lab in column_indices])
                    batched_labels.append(labels_batch)
            else:
                for batch in batches_indices:
                    # Use slicing since document_data is a tensor
                    batched_input.append(document_data[batch])
                    
                    labels_batch = []
                    for idx in batch:
                        labels_batch.append([0] + [self.cfg['task_to_subroot'][task_id]] + [lab for lab in labels_data[idx] if lab in column_indices])
                    batched_labels.append(labels_batch)

            return (
                batched_input,
                batched_labels,
                column_indices,
            )
        return _collate_tamlec
