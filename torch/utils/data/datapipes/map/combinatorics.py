import random

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import MapDataPipe
from typing import Iterator, List, Optional, TypeVar

__all__ = ["ShufflerMapDataPipe", ]


T_co = TypeVar('T_co', covariant=True)


@functional_datapipe('shuffle')
class ShufflerMapDataPipe(MapDataPipe[T_co]):
    r"""
    Shuffle the input DataPipe via its indices (functional name: ``shuffle``).

    When it is used with :class:`~torch.utils.data.DataLoader`, the methods to
    set up random seed are different based on :attr:`num_workers`.

    For single-process mode (:attr:`num_workers == 0`), the random seed is set before
    the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
    mode (:attr:`num_worker > 0`), ``worker_init_fn`` is used to set up a random seed
    for each worker process.

    Args:
        datapipe: MapDataPipe being shuffled
        indices: a list of indices of the MapDataPipe. If not provided, we assume it uses 0-based indexing

    Example:
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> shuffle_dp = dp.shuffle()
        >>> list(shuffle_dp)
        [0, 4, 1, 6, 3, 2, 9, 5, 7, 8]
    """
    datapipe: MapDataPipe[T_co]

    def __init__(self,
                 datapipe: MapDataPipe[T_co],
                 *,
                 indices: Optional[List] = None,
                 ) -> None:
        super().__init__()
        self.datapipe = datapipe
        self.indices = list(range(len(datapipe))) if indices is None else indices
        self.index_map = {index_name: num_index for num_index, index_name in enumerate(self.indices)}
        self._enabled = True
        self._seed = None
        self._rng = random.Random()
        self._reset = True
        self.shuffled_indices = None

    def set_shuffle(self, shuffle=True):
        self._enabled = shuffle
        return self

    def set_seed(self, seed: int):
        self._seed = seed
        self._reset = True
        return self

    def __getitem__(self, index) -> T_co:
        try:
            old_numeric_index = self.index_map[index]
        except KeyError:
            raise IndexError(f"Index {index} is out of range for {self}.")

        if self._enabled:
            if self._reset:
                if self._seed is None:
                    self._seed = int(torch.empty((), dtype=torch.int64).random_().item())
                self._rng.seed(self._seed)
                self.shuffled_indices = self._rng.sample(self.indices)
                self._reset = False
            new_index = self.shuffled_indices[old_numeric_index]
        else:
            new_index = self.indices[old_numeric_index]
        return self.datapipe[new_index]

    # Without __iter__ implemented, by default it tries to use 0-index,
    # which doesn't work when there is a custom index.
    def __iter__(self) -> Iterator[T_co]:
        for i in self.indices:
            yield self.datapipe[i]

    def __len__(self) -> int:
        return len(self.datapipe)
