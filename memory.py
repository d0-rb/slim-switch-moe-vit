# simple memory to store past samples for rehearsal
import torch


class RehearsalMemory:
    def __init__(self, max_size, input_shape, output_shape, device):
        self.size = 0
        self.max_size = max_size
        self.device = device
        self._batch = torch.empty((self.max_size, *input_shape), device=device)
        self._labels = torch.empty((self.max_size, *output_shape), device=device)

    
    # randomly select num_samples from batch tensor to add into memory. if memory is full, randomly replace some existing samples
    def add(self, batch: torch.Tensor, labels: torch.Tensor, num_samples: int):
        assert num_samples <= batch.shape[0], 'number of samples to save more than batch size!'
        assert num_samples <= labels.shape[0], 'number of samples to save more than labels size!'
        assert batch.shape[0] == labels.shape[0], 'batch size does not match labels size!'

        idx = torch.randperm(batch.shape[0], device=self.device)[:num_samples]  # indexes of randomly selected samples from batch
        samples = batch[idx]  # randomly selected samples from batch to be saved
        sample_labels = labels[idx]  # corresponding labels
        
        if self.size + self.num_samples > self.max_size:  # if we are going to overfill memory
            free_space = self.max_size - self.size
            self._batch[self.size:self.size + free_space] = samples[:free_space]
            self._labels[self.size:self.size + free_space] = sample_labels[:free_space]
            samples = samples[free_space:]
            sample_labels = sample_labels[free_space:]
        
            num_replacements = num_samples - free_space  # number of samples that will have to replace other samples
            replaced_idx = torch.randperm(self.max_size, device=self.device)[:num_replacements]  # indexes of samples in batch to be replaced
            self._batch[replaced_idx] = samples
            self._labels[replaced_idx] = sample_labels
        else:  # if we have enough space to store all the samples we need
            self._batch[self.size:self.size + num_samples] = samples
            self._labels[self.size:self.size + num_samples] = sample_labels
        
        self.size = min(self.max_size, self.size + self.num_samples)


    @property
    def batch(self):
        return self._batch[:self.size]
    
    @property
    def labels(self):
        return self._labels[:self.size]
