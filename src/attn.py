"""
We implement the “Attention Layer” proposed by Luong et al. in 2015 as a separate nn.Module called Attn.
"""

import torch
from torch import nn

class Attn(torch.nn.Module):
  """
  The output of this module is a softmax normalized weights tensor of shape (batch_size, 1, max_length).
  """

  def __init__(self, method, hidden_size):
    super(Attn, self).__init__()
    self.hidden_size = hidden_size
    self.method = method
    if self.method not in ['dot', 'general', 'concat']:
      raise ValueError(self.method, 'is not an appropriate attention method.')
    if self.method == 'general':
      self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
    elif self.method == 'concat':
      self.attn = torch.nn.Linear(self.hidden_size*2, hidden_size)
      self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

  def dot_score(self, hidden, encoder_output):
	# shape = (max_length, batch_size)
    return torch.sum(hidden*encoder_output, dim=2)

  def general_score(self, hidden, encoder_output):
	# shape = (max_length, batch_size, hidden_size)
    energy = self.attn(encoder_output)
	# shape = (max_length, batch_size)
    return torch.sum(hidden*energy, dim=2)

  def concat_score(self, hidden, encoder_output):
	# shape = (max_length, batch_size, hidden_size)
    energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
	# shape = (max_length, batch_size)
    return torch.sum(self.v*energy, dim=2)

  def forward(self, hidden, encoder_outputs):
    # Calculate the attention weights (energies) based on the given method
    if self.method == 'general':
      attn_energies = self.general_score(hidden, encoder_outputs)
    elif self.method == 'concat':
      attn_energies = self.concat_score(hidden, encoder_outputs)
    elif self.method == 'dot':
      attn_energies = self.dot_score(hidden, encoder_outputs)

    # Transpose max_length and batch_size dimensions
	# shape = (batch_size, max_length)
    attn_energies = attn_energies.t()

    # Return the softmax normalized probability scores (with added dimension)
	# shape = (batch_size, 1, max_length)
    return nn.functional.softmax(attn_energies, dim=1).unsqueeze(1)
