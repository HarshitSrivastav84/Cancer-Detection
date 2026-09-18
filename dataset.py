import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_TO_IDX = {
    "non_cancer": 0,
    "cancer": 1
}

# Preprocessing pipelines
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

# Validation/Test
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])


class GastricDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        """
        csv_path: path to dataset_labels.csv
        split: one of "train", "val", "test"
        transform: which preprocessing pipeline to apply
        """
        full_df = pd.read_csv(csv_path)
        self.data = full_df[full_df["split"] == split].reset_index(drop=True)
        self.transform = transform

        print(f"Loaded {split} split: {len(self.data)} images "
              f"({self.data['label'].value_counts().to_dict()})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row["image_path"]
        label_text = row["label"]
        label = LABEL_TO_IDX[label_text]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Quick test when running this file directly
if __name__ == "__main__":
    print("Testing dataset loading...\n")

    train_dataset = GastricDataset(
        csv_path="dataset_labels.csv",
        split="train",
        transform=train_transform
    )

    val_dataset = GastricDataset(
        csv_path="dataset_labels.csv",
        split="val",
        transform=eval_transform
    )

    test_dataset = GastricDataset(
        csv_path="dataset_labels.csv",
        split="test",
        transform=eval_transform
    )

    print(f"\nTotal train: {len(train_dataset)}")
    print(f"Total val:   {len(val_dataset)}")
    print(f"Total test:  {len(test_dataset)}")

    # Load a single sample and check its shape
    image, label = train_dataset[0]
    print(f"\nSample image tensor shape: {image.shape}")
    print(f"Sample label: {label} (0=non_cancer, 1=cancer)")
    print("\nDataset loading works correctly!")