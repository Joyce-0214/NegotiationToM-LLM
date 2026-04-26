# -*- coding: utf-8 -*-
"""
torchtext 兼容层
替代已废弃的 torchtext.data.Dataset, Field, Example, Iterator 等

适用于 PyTorch 2.x + torchtext 0.15+
"""

import torch
from torch.utils.data import Dataset as TorchDataset
from collections import Counter, OrderedDict
from itertools import chain


class Vocab:
    """简化版 Vocab，替代 torchtext.vocab.Vocab"""

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<unk>'],
                 specials_first=True):
        self.freqs = counter
        self.itos = []
        self.stoi = {}

        if specials_first:
            self.itos = list(specials)

        # 按频率排序
        words_and_freqs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        for word, freq in words_and_freqs:
            if freq < min_freq:
                continue
            if max_size is not None and len(self.itos) >= max_size:
                break
            if word not in specials:
                self.itos.append(word)

        if not specials_first:
            self.itos.extend(specials)

        self.stoi = {word: i for i, word in enumerate(self.itos)}
        self.unk_index = self.stoi.get('<unk>', 0)

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, self.unk_index)


class Example:
    """替代 torchtext.data.Example"""

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        for key, val in data.items():
            if key in fields and fields[key] is not None:
                setattr(ex, key, fields[key].preprocess(val))
            else:
                setattr(ex, key, val)
        return ex

    @classmethod
    def fromlist(cls, data, fields):
        ex = cls()
        for (name, field), val in zip(fields, data):
            if field is not None:
                setattr(ex, name, field.preprocess(val))
            else:
                setattr(ex, name, val)
        return ex


class RawField:
    """不做任何处理的 Field"""

    def __init__(self, preprocessing=None, postprocessing=None):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, x):
        if self.preprocessing is not None:
            return self.preprocessing(x)
        return x

    def process(self, batch, *args, **kwargs):
        if self.postprocessing is not None:
            return self.postprocessing(batch)
        return batch


class Field(RawField):
    """替代 torchtext.data.Field"""

    def __init__(self, sequential=True, use_vocab=True, init_token=None,
                 eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False,
                 tokenize=None, tokenizer_language='en', include_lengths=False,
                 batch_first=False, pad_token='<pad>', unk_token='<unk>',
                 pad_first=False, truncate_first=False, stop_words=None,
                 is_target=False):
        super().__init__(preprocessing, postprocessing)

        self.sequential = sequential
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.lower = lower
        self.tokenize = tokenize if tokenize is not None else (lambda x: x.split())
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        self.stop_words = stop_words or set()
        self.is_target = is_target

        self.vocab = None

    def preprocess(self, x):
        if self.sequential and isinstance(x, str):
            x = self.tokenize(x)
        if self.lower:
            x = [w.lower() for w in x] if self.sequential else x.lower()
        if self.sequential and self.stop_words:
            x = [w for w in x if w not in self.stop_words]
        if self.preprocessing is not None:
            x = self.preprocessing(x)
        return x

    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        for data in args:
            for example in data:
                if hasattr(example, self.name if hasattr(self, 'name') else ''):
                    val = getattr(example, self.name)
                else:
                    val = example
                if self.sequential:
                    counter.update(val)
                else:
                    counter[val] += 1

        specials = []
        if self.pad_token is not None:
            specials.append(self.pad_token)
        if self.unk_token is not None:
            specials.append(self.unk_token)
        if self.init_token is not None:
            specials.append(self.init_token)
        if self.eos_token is not None:
            specials.append(self.eos_token)

        self.vocab = Vocab(counter, specials=specials, **kwargs)

    def pad(self, minibatch):
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch

        if self.fix_length is not None:
            max_len = self.fix_length
        else:
            max_len = max(len(x) for x in minibatch)

        if self.init_token is not None:
            max_len += 1
        if self.eos_token is not None:
            max_len += 1

        padded = []
        lengths = []
        for x in minibatch:
            if self.init_token is not None:
                x = [self.init_token] + list(x)
            if self.eos_token is not None:
                x = list(x) + [self.eos_token]

            lengths.append(len(x))

            if self.pad_first:
                x = [self.pad_token] * (max_len - len(x)) + list(x)
            else:
                x = list(x) + [self.pad_token] * (max_len - len(x))

            if self.truncate_first:
                x = x[-max_len:]
            else:
                x = x[:max_len]

            padded.append(x)

        if self.include_lengths:
            return padded, lengths
        return padded

    def numericalize(self, arr, device=None):
        if self.include_lengths:
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=torch.long, device=device)

        if self.use_vocab:
            if self.sequential:
                arr = [[self.vocab[x] for x in ex] for ex in arr]
            else:
                arr = [self.vocab[x] for x in arr]

        var = torch.tensor(arr, dtype=self.dtype, device=device)

        if self.sequential and not self.batch_first:
            var = var.t().contiguous()

        if self.include_lengths:
            return var, lengths
        return var

    def process(self, batch, device=None):
        padded = self.pad(batch)
        return self.numericalize(padded, device=device)


class NestedField(Field):
    """替代 torchtext.data.NestedField"""

    def __init__(self, nesting_field, use_vocab=True, init_token=None,
                 eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, tokenize=None,
                 pad_token='<pad>', pad_first=False, include_lengths=False):
        super().__init__(
            use_vocab=use_vocab,
            init_token=init_token,
            eos_token=eos_token,
            fix_length=fix_length,
            dtype=dtype,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            tokenize=tokenize,
            pad_token=pad_token,
            pad_first=pad_first,
            include_lengths=include_lengths
        )
        self.nesting_field = nesting_field


class Dataset(TorchDataset):
    """替代 torchtext.data.Dataset"""

    def __init__(self, examples, fields, filter_pred=None):
        if filter_pred is not None:
            examples = list(filter(filter_pred, examples))

        self.examples = examples

        # fields 可以是 dict 或 list of tuples
        if isinstance(fields, dict):
            self.fields = fields
        else:
            self.fields = dict(fields)

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        for ex in self.examples:
            yield ex

    def __getattr__(self, attr):
        if attr in self.fields:
            for ex in self.examples:
                yield getattr(ex, attr)

    @classmethod
    def splits(cls, path=None, root='.data', train=None, validation=None,
               test=None, **kwargs):
        """创建训练/验证/测试集分割"""
        datasets = []
        for name in (train, validation, test):
            if name is not None:
                datasets.append(cls(name, **kwargs))
            else:
                datasets.append(None)
        return tuple(d for d in datasets if d is not None)


class Batch:
    """替代 torchtext.data.Batch"""

    def __init__(self, data, dataset, device=None):
        self.batch_size = len(data)
        self.dataset = dataset

        for (name, field) in dataset.fields.items():
            if field is not None:
                batch_data = [getattr(x, name) for x in data]
                setattr(self, name, field.process(batch_data, device=device))


class Iterator:
    """替代 torchtext.data.Iterator"""

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None, train=True, repeat=False, shuffle=None,
                 sort=None, sort_within_batch=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sort_key = sort_key
        self.device = device
        self.batch_size_fn = batch_size_fn
        self.train = train
        self.repeat = repeat
        self.shuffle = shuffle if shuffle is not None else train
        self.sort = sort if sort is not None else not train
        self.sort_within_batch = sort_within_batch

        self._iterations = 0

    def __iter__(self):
        while True:
            self._iterations += 1
            indices = list(range(len(self.dataset)))

            if self.shuffle:
                import random
                random.shuffle(indices)
            elif self.sort and self.sort_key is not None:
                indices.sort(key=lambda i: self.sort_key(self.dataset[i]))

            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_data = [self.dataset[j] for j in batch_indices]

                if self.sort_within_batch and self.sort_key is not None:
                    batch_data.sort(key=self.sort_key, reverse=True)

                yield Batch(batch_data, self.dataset, self.device)

            if not self.repeat:
                break

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class BucketIterator(Iterator):
    """替代 torchtext.data.BucketIterator"""

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None, train=True, repeat=False, shuffle=None,
                 sort=None, sort_within_batch=None):
        super().__init__(
            dataset, batch_size, sort_key=sort_key, device=device,
            batch_size_fn=batch_size_fn, train=train, repeat=repeat,
            shuffle=shuffle, sort=sort, sort_within_batch=sort_within_batch
        )


def batch(data, batch_size, batch_size_fn=None):
    """替代 torchtext.data.batch"""
    minibatch = []
    for ex in data:
        minibatch.append(ex)
        if len(minibatch) == batch_size:
            yield minibatch
            minibatch = []
    if minibatch:
        yield minibatch


def pool(data, batch_size, key, batch_size_fn=None, random_shuffler=None):
    """替代 torchtext.data.pool"""
    for p in batch(data, batch_size * 100, batch_size_fn):
        p_sorted = sorted(p, key=key)
        for b in batch(p_sorted, batch_size, batch_size_fn):
            yield b
