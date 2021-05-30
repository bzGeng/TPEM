import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import _VF
import numpy as np
from torch._jit_internal import _parameter_list
from torch.nn.parameter import Parameter
import torch.nn.init as init
from torch.nn.utils.rnn import PackedSequence
from copy import deepcopy
import warnings

_rnn_impls = {
    'RNN_TANH': _VF.rnn_tanh,
    'RNN_RELU': _VF.rnn_relu,
}

DEFAULT_THRESHOLD = 5e-3


class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    @staticmethod
    def forward(ctx, inputs, threshold):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1
        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


class SharableEmbedding(nn.Module):
    """Modified linear layer."""
    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None, scale=0.5):
        super(SharableEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.scale = scale

        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(_weight)
        self.sparse = sparse

        self.piggymask = None
        self.threshold_fn = Binarizer.apply

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -self.scale, self.scale)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input):
        if self.piggymask is not None:
            mask_thresholded = self.threshold_fn(self.piggymask, DEFAULT_THRESHOLD)
            weight = mask_thresholded * self.weight
        else:
            weight = self.weight
        return F.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def __repr__(self):
        s = self.__class__.__name__ + '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None,
                        max_norm=None, norm_type=2., scale_grad_by_freq=False,
                        sparse=False):
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse)
        embedding.weight.requires_grad = not freeze
        return embedding


def apply_permutation(tensor, permutation, dim=1):
    # type: (Tensor, Tensor, int) -> Tensor
    return tensor.index_select(dim, permutation)


class SharableRNNBase(nn.Module):
    __constants__ = ['mode', 'input_size', 'hidden_size', 'num_layers', 'bias',
                     'batch_first', 'dropout', 'bidirectional']

    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0., bidirectional=False):
        super(SharableRNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        elif mode == 'RNN_TANH':
            gate_size = hidden_size
        elif mode == 'RNN_RELU':
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self.threshold_fn = Binarizer.apply

        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                # Second bias vector included for CuDNN compatibility. Only one
                # bias vector is needed in standard definition.
                b_hh = Parameter(torch.Tensor(gate_size))

                layer_params = (w_ih, w_hh, b_ih, b_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

            self.piggymasks = [p+'_piggymask' for p in [p for layerparams in self._all_weights  for p in layerparams] if 'weight' in p]
            self.piggymasks_weights = {name: None for name in self.piggymasks}
        self.flatten_parameters()
        self.reset_parameters()

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        all_weights = self._flat_weights
        unique_data_ptrs = set(p.data_ptr() for p in all_weights)
        if len(unique_data_ptrs) != len(all_weights):
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn

            # NB: This is a temporary hack while we still don't have Tensor
            # bindings for ATen functions
            with torch.no_grad():
                # NB: this is an INPLACE function on all_weights, that's why the
                # no_grad() is necessary.
                torch._cudnn_rnn_flatten_weight(
                    all_weights, (4 if self.bias else 2),
                    self.input_size, rnn.get_cudnn_mode(self.mode), self.hidden_size, self.num_layers,
                    self.batch_first, bool(self.bidirectional))

    def _apply(self, fn):
        ret = super(SharableRNNBase, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in dict(self.named_parameters()).items():
            if len(weight.size()) == 1:
                weight.data.zero_()
                continue
            init.uniform_(weight, -stdv, stdv)

    def _get_flat_weights_names(self):
        return [weight for weights in self._all_weights for weight in weights]

    @_parameter_list(_get_flat_weights_names)
    def _get_flat_weights(self):
        return self._flat_weights

    def check_input(self, input, batch_sizes):
        # type: (Tensor, Optional[Tensor]) -> None
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

    def get_expected_hidden_size(self, input, batch_sizes):
        # type: (Tensor, Optional[Tensor]) -> Tuple[int, int, int]
        if batch_sizes is not None:
            mini_batch = batch_sizes[0]
            mini_batch = int(mini_batch)
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        return expected_hidden_size

    def check_hidden_size(self, hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
        # type: (Tensor, Tuple[int, int, int], str) -> None
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

    def check_forward_args(self, input, hidden, batch_sizes):
        # type: (Tensor, Tensor, Optional[Tensor]) -> None
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    def permute_hidden(self, hx, permutation):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        if permutation is None:
            return hx
        return apply_permutation(hx, permutation)

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=input.dtype, device=input.device)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        _impl = _rnn_impls[self.mode]
        if batch_sizes is None:
            result = _impl(input, hx, self._get_flat_weights(), self.bias, self.num_layers,
                           self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _impl(input, batch_sizes, hx, self._get_flat_weights(), self.bias,
                           self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1]

        if is_packed:
            output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return output, self.permute_hidden(hidden, unsorted_indices)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(SharableRNNBase, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:2]]

    def flat_and_piggy_weights(self):
        if self.piggymasks_weights[getattr(self, 'piggymasks')[0]] is not None:
            input_weights = []
            for name in [p for layerparams in self._all_weights for p in layerparams]:
                if 'weight' in name:
                    name_piggy = name + '_piggymask'
                    mask_thresholded = self.threshold_fn(self.piggymasks_weights[name_piggy], DEFAULT_THRESHOLD)
                    input_weights += [getattr(self, name) * mask_thresholded]
                else:
                    input_weights += [getattr(self, name)]
            self.flatten_parameters_for_piggy(input_weights)
            return input_weights
        else:
            return self._flat_weights

    def flatten_parameters_for_piggy(self, sampled_weights):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = sampled_weights[0].data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        all_weights = sampled_weights
        unique_data_ptrs = set(p.data_ptr() for p in all_weights)
        if len(unique_data_ptrs) != len(all_weights):
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn

            # NB: This is a temporary hack while we still don't have Tensor
            # bindings for ATen functions
            with torch.no_grad():
                # NB: this is an INPLACE function on all_weights, that's why the
                # no_grad() is necessary.
                torch._cudnn_rnn_flatten_weight(
                    all_weights, (4 if self.bias else 2),
                    self.input_size, rnn.get_cudnn_mode(self.mode), self.hidden_size, self.num_layers,
                    self.batch_first, bool(self.bidirectional))

    @property
    def _flat_weights(self):
        return [p for layerparams in self.all_weights for p in layerparams]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]


class SharableGRU(SharableRNNBase):
    """Modified linear layer."""
    __overloads__ = {'forward': ['forward_packed', 'forward_tensor']}

    def __init__(self, *args, **kwargs):
        super(SharableGRU, self).__init__('GRU', *args, **kwargs)

    def run_impl(self, input, hx, batch_sizes):
        if batch_sizes is None:
            result = _VF.gru(input, hx, self.flat_and_piggy_weights(), self.bias, self.num_layers,
                             self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.gru(input, batch_sizes, hx, self.flat_and_piggy_weights(), self.bias,
                             self.num_layers, self.dropout, self.training, self.bidirectional)
        return result

    def forward_impl(self, input, hx, batch_sizes, max_batch_size, sorted_indices):
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=input.dtype, device=input.device)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        result = self.run_impl(input, hx, batch_sizes)
        output = result[0]
        hidden = result[1]
        return output, hidden

    @torch._jit_internal.export
    def forward_packed(self, input, hx=None):
        input, batch_sizes, sorted_indices, unsorted_indices = input
        max_batch_size = batch_sizes[0]
        max_batch_size = int(max_batch_size)
        output, hidden = self.forward_impl(input, hx, batch_sizes, max_batch_size, sorted_indices)
        output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return output, self.permute_hidden(hidden, unsorted_indices)

    @torch._jit_internal.export
    def forward_tensor(self, input, hx=None):
        batch_sizes = None
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        sorted_indices = None
        unsorted_indices = None
        output, hidden = self.forward_impl(input, hx, batch_sizes, max_batch_size, sorted_indices)
        return output, self.permute_hidden(hidden, unsorted_indices)

    @torch._jit_internal.ignore
    def forward(self, input, hx=None):
        if isinstance(input, PackedSequence):
            return self.forward_packed(input, hx)
        else:
            return self.forward_tensor(input, hx)


class SharableLinear(nn.Module):
    """Modified linear layer."""
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(SharableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight and bias are no longer Parameters.
        self.weight = Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        nn.init.normal_(self.weight, 0, 0.01)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features), requires_grad=True)
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        self.piggymask = None
        self.threshold_fn = Binarizer.apply

    def forward(self, input):
        if self.piggymask is not None:
            mask_thresholded = self.threshold_fn(self.piggymask, DEFAULT_THRESHOLD)
            weight = mask_thresholded * self.weight
        else:
            weight = self.weight

        # Get output using modified weight.
        return F.linear(input, weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) + ')'

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        if self.bias is not None and self.bias.data is not None:
            self.bias.data = fn(self.bias.data)
