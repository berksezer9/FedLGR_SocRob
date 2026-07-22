from torch.utils.data import Dataset
import torch
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None, label_start=4):
        # label_start=4 reproduces original MANNERS-DB layout: Stamp, path,
        # Using circle, Using arrow, then 8 action columns. OfficeDB adaptation
        # (OFFICEDB_MODIFICATIONS.md) passes a different count of leading
        # id/group/task/split columns, so callers compute label_start as
        # 2 + len(extra_cols).
        self.dataframe = dataframe
        self.transform = transform
        self.label_start = label_start

    def __len__(self):
        return len(self.dataframe)

    def __get_dataframe__(self):
        return self.dataframe

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 1]
        image = Image.open(img_path).convert('RGB')
        image = image.crop((295, 0, 295+1018, image.size[1]))

        labels = torch.tensor(self.dataframe.iloc[idx, self.label_start:].values.astype('float32'), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels