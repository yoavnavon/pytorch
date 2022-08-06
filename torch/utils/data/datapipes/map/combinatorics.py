import random

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import MapDataPipe
from typing import Iterator, List, Optional, TypeVar, Union
from copy import deepcopy

__all__ = ["ShufflerMapDataPipe", "DropperDataPipe", ]


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
        # We do not lazily shuffle because this way is significantly faster in terms of total time
        random.shuffle(self.indices)

    def __getitem__(self, index) -> T_co:
        try:
            old_numeric_index = self.index_map[index]
        except KeyError:
            raise IndexError(f"Index {index} is out of range for {self}.")
        new_index = self.indices[old_numeric_index]
        return self.datapipe[new_index]

    # Without __iter__ implemented, by default it tries to use 0-index,
    # which doesn't work when there is a custom index.
    def __iter__(self) -> Iterator[T_co]:
        for i in self.indices:
            yield self.datapipe[i]

    def __len__(self) -> int:
        return len(self.datapipe)

@functional_datapipe('drop')
class DropperMapDataPipe(MapDataPipe[T_co]):
    r"""
    Drop columns/elements in input DataPipe via its indices (functional name: ``drop``).

    Args:
        datapipe: MapDataPipe with columns to be dropped
        indices: a single column index to be dropped or a list of indices

    Example:
        >>> from torchdata.datapipes.map import SequenceWrapper, ZipperMapDataPipe
        >>> dp1 = SequenceWrapper(range(5))
        >>> dp2 = SequenceWrapper(range(10, 15))
        >>> dp = dp1.zip(dp2)
        >>> list(dp)
        [(0, 10), (1, 11), (2, 12), (3, 13), (4, 14)]
        >>> drop_dp = dp.drop(1)
        >>> list(drop_dp)
        [(0), (1), (2), (3), (4)]
    """
    datapipe: MapDataPipe[T_co]

    def __init__(self,
                 datapipe: MapDataPipe[T_co],
                 indices: Union[int, List],
                 ) -> None:
        super().__init__()
        self.datapipe = datapipe
        if isinstance(indices, list):
            self.indices = indices
        else:
            self.indices = [indices]

    def __getitem__(self, index) -> T_co:
        old_item = self.datapipe[index]
        new_item = deepcopy(old_item)
        if isinstance(old_item, tuple):
            new_item = list(new_item)
        
        for i in self.indices:
            del new_item[i]
        
        if isinstance(old_item, tuple):
            new_item = tuple(new_item)

        return new_item

    def __len__(self) -> int:
        return len(self.datapipe)
