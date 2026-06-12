# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union, Any

from mmengine import fileio
from mmengine.logging import MMLogger

from mmpretrain.registry import DATASETS
from .custom import CustomDataset

CELL_CATEGORIES = ('erythrocyte-1-1',
                'erythrocyte-2-1',
                'erythrocyte-3-1',
                'erythrocyte-3-2-1',
                'erythrocyte-4-1',
                'erythrocyte-4-2-2',
                'erythrocyte-6-1',
                'erythrocyte-6-2-10',
                'erythrocyte-6-2-20',
                'erythrocyte-6-2-6',
                'erythrocyte-6-2-8',
                'erythrocyte-6-2-9',
                'erythrocyte-6-4-4',
                'granulocyte-1-1',
                'granulocyte-2-1',
                'granulocyte-2-2-0',
                'granulocyte-3-1',
                'granulocyte-4-1',
                'granulocyte-5-1',
                'granulocyte-6-1',
                'granulocyte-7-1',
                'granulocyte-7-2',
                'granulocyte-7-3',
                'granulocyte-7-4',
                'granulocyte-8-1',
                'granulocyte-8-2',
                'granulocyte-8-3',
                'granulocyte-8-4',
                'lymphocyte-1-1',
                'lymphocyte-2',
                'lymphocyte-3-1',
                'megakaryocyte-1-1',
                'megakaryocyte-1-2',
                'megakaryocyte-1-3',
                'megakaryocyte-1-4',
                'megakaryocyte-1-5',
                'monocyte-1',
                'monocyte-2',
                'monocyte-3',
                'othercells-2',
                'plasmacells-1',
                'plasmacells-2',
                'plasmacells-3',
                'plasmacells-4-5',
                'platelet-1-1',
                'platelet-1-2',
                'platelet-1-5',
                )


@DATASETS.register_module()
class BCFC_329k(CustomDataset):
    """`ImageNet <http://www.image-net.org>`_ Dataset.

    The dataset supports two kinds of directory format,

    ::

        imagenet
        ├── train
        │   ├──class_x
        |   |   ├── x1.jpg
        |   |   ├── x2.jpg
        |   |   └── ...
        │   ├── class_y
        |   |   ├── y1.jpg
        |   |   ├── y2.jpg
        |   |   └── ...
        |   └── ...
        ├── val
        │   ├──class_x
        |   |   └── ...
        │   ├── class_y
        |   |   └── ...
        |   └── ...
        └── test
            ├── test1.jpg
            ├── test2.jpg
            └── ...

    or ::

        imagenet
        ├── train
        │   ├── x1.jpg
        │   ├── y1.jpg
        │   └── ...
        ├── val
        │   ├── x3.jpg
        │   ├── y3.jpg
        │   └── ...
        ├── test
        │   ├── test1.jpg
        │   ├── test2.jpg
        │   └── ...
        └── meta
            ├── train.txt
            └── val.txt


    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        split (str): The dataset split, supports "train", "val" and "test".
            Default to ''.
        data_prefix (str | dict): Prefix for training data. Defaults to ''.
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        **kwargs: Other keyword arguments in :class:`CustomDataset` and
            :class:`BaseDataset`.


    Examples:
        >>> from mmpretrain.datasets import ImageNet
        >>> train_dataset = ImageNet(data_root='data/imagenet', split='train')
        >>> train_dataset
        Dataset ImageNet
            Number of samples:  1281167
            Number of categories:       1000
            Root of dataset:    data/imagenet
        >>> test_dataset = ImageNet(data_root='data/imagenet', split='val')
        >>> test_dataset
        Dataset ImageNet
            Number of samples:  50000
            Number of categories:       1000
            Root of dataset:    data/imagenet
    """  # noqa: E501

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    METAINFO = {'classes': CELL_CATEGORIES}

    def __init__(self,
                 data_root: str = '',
                 split: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 **kwargs):
        kwargs = {'extensions': self.IMG_EXTENSIONS, **kwargs}

        if split:
            splits = ['train', 'val', 'test']
            assert split in splits, \
                f"The split must be one of {splits}, but get '{split}'"

            data_prefix = split if data_prefix == '' else data_prefix

            if ann_file == '':
                _ann_path = fileio.join_path(data_root, 'meta', f'{split}.txt')
                if fileio.exists(_ann_path):
                    ann_file = fileio.join_path('meta', f'{split}.txt')

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            metainfo=metainfo,
            **kwargs)

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body
    
    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        # debug = self.pipeline(data_info)
        return self.pipeline(data_info)