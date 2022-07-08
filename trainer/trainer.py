import collections
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import Trainer
from typing import Optional

class TrainerWithEvalCollator(Trainer):
    def __init__(self, *args, **kwargs):
        if "eval_data_collator" in kwargs:
            self.eval_data_collator = kwargs["eval_data_collator"]
            del kwargs["eval_data_collator"]
        else:
            self.eval_data_collator = None

        super(TrainerWithEvalCollator, self).__init__(*args, **kwargs)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if self.eval_data_collator is None:
            return super().get_eval_dataloader(eval_dataset)

        # Guards borrowed from base Trainer
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("TrainerWithEvalCollator: evaluation requires an eval_dataset.")
        elif eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.eval_data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )
