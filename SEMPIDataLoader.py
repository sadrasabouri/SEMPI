# %%
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Any
from itertools import combinations, chain
import os

PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PATH, 'data')

# %%
def all_subsets(lst: List[Any]) -> List[List[Any]]:
    return list(chain(*[combinations(lst, i) for i in range(len(lst) + 1)]))

all_subsets([1, 2, 3])

# %%
META_DATA_COLUMNS = ['frame', 'face_id', 'timestamp', 'confidence', 'success']

class InterPersenSEMPIDataset(Dataset):
    """
    Dataset class for the InterPersenSEMPI dataset.

    Args:
        features (List[List[pd.DataFrame]]): A list of containing a dictionary of features for each person, starting from the person who is predicant 
        engagements (List[float]): List of engagements as floats
        frame_length (int): Frame length of the features
    """
    def __init__(self,
                 features: List[List[pd.DataFrame]],
                 engagements: List[float],
                 pids: List[List[int]],
                 frame_length: int = 64):
        self.features = features
        self.engagements = engagements
        self.pids = pids
        self.frame_length = frame_length

    def __len__(self):
        return len(self.engagements)

    def _get_features(self, features: List[pd.DataFrame]) -> np.ndarray:
        """
        Construct (n_persons, n_features, n_frames) from features and zero-pad/cut if necessary.
        """
        n_persons = len(features)
        n_features = len(features[0].columns) - len(META_DATA_COLUMNS)
        n_frames = self.frame_length

        # Construct the feature tensor
        feature_tensor = np.zeros((n_persons, n_features, n_frames))
        for i, feature in enumerate(features):
            feature = feature.loc[:, ~feature.columns.isin(META_DATA_COLUMNS)].values
            feature = feature[:n_frames, :]
            feature_tensor[i, :, :feature.shape[0]] = feature.T

        assert feature_tensor.shape == (n_persons, n_features, n_frames)
        return feature_tensor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        features: List[pd.DataFrame] = self.features[idx]
        engagement: float  = self.engagements[idx]
        pids = self.pids[idx]

        item = {}
        item['score'] = torch.tensor(engagement, dtype=torch.float32)
        item['features'] = torch.tensor(self._get_features(features), dtype=torch.float32)
        item['pids'] = torch.tensor(pids, dtype=torch.int32)
        return item
    
    def get_metadata(self, idx: int) -> Dict[str, Any]:
        features: List[pd.DataFrame] = self.features[idx]
        pids: List[int] = self.pids[idx]
        result = {
            f"pid_{pid}": {
                c: features[i].loc[:, c].tolist() for c in META_DATA_COLUMNS
            } for i, pid in enumerate(pids)
        }
        result['pids'] = pids
        return result

# %%
class DataSetLoader():
    """
    Load the dataset from disk and create a dataset object.

    Args:
        data_path (str): Path to the dataset
    """
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
    
    def _load_engagement(self) -> pd.DataFrame:
        """Load the labels from the dataset."""
        engagement_dfs = []
        for fold in [os.path.join(DATA_PATH, "engagement", f"label_0402_fold_{i}") for i in range(5)]:
            train_df = pd.read_csv(f"{fold}/train.csv")
            val_df = pd.read_csv(f"{fold}/val.csv")
            engagement_dfs.extend([train_df, val_df])
        engagement_dfs = pd.concat(engagement_dfs, ignore_index=True)
        engagement_dfs = engagement_dfs.sort_values(by='video_path', ascending=True)
        engagement_dfs['filename'] = engagement_dfs['video_path'].apply(lambda x: x.split('/')[-1] + ".csv")
        return engagement_dfs

    def _load_openface_features(self, engagement_df: pd.DataFrame) -> Tuple[List[float], List[List[pd.DataFrame]]]:
        """Load the OpenFace features from the dataset."""
        video_paths_set = set(engagement_df["filename"])
        BASE_PATH = os.path.join(DATA_PATH, "engagement", "featopenface")
        
        openface_features = []
        engagements = []
        person_ids = []
        max_df_len = 0
        for video_folder in os.listdir(BASE_PATH):
            video_folder_path = os.path.join(BASE_PATH, video_folder)
            if not os.path.isdir(video_folder_path):
                continue
            print(f"Processing folder: {video_folder}")

            for clip_folder in os.listdir(video_folder_path):
                clip_path = os.path.join(video_folder_path, clip_folder)
                if not os.path.isdir(clip_path):
                    continue

                person_dfs = {}
                for person_df_fname in os.listdir(clip_path):
                    if person_df_fname in video_paths_set:
                        person_df_path = os.path.join(clip_path, person_df_fname)
                        try:
                            df = pd.read_csv(person_df_path)
                            if len(df.index) > max_df_len:
                                max_df_len = len(df.index)
                            engagement_value = engagement_df.loc[engagement_df["filename"] == person_df_fname,
                                                                 "engagement"].values[0]
                            person_id = int(person_df_fname.split('.')[0].split('_')[-1][-1])
                            person_dfs[person_id] = {
                                "df": df,
                                "engagement": engagement_value
                            }

                        except Exception as e:
                            print(f"Could not read {person_df_fname}: {e}")

                if len(person_dfs) < 2:
                    continue

                for (p0, data0), (p1, data1) in combinations(person_dfs.items(), 2): # TODO: Change it to all subsets starting with pi for each pi (pi + comb(A - pi))
                    e0, df0 = data0["engagement"], data0["df"]
                    e1, df1 = data1["engagement"], data1["df"]
                    if df0.equals(df1):
                        continue

                    engagements.append(e0)
                    openface_features.append([df0, df1])
                    person_ids.append([p0, p1])

                    engagements.append(e1)
                    openface_features.append([df1, df0])
                    person_ids.append([p1, p0])
        return engagements, openface_features, person_ids


    def get_dataset(self) -> InterPersenSEMPIDataset:
        engagement_df = self._load_engagement()
        engagements, openface_features, person_ids = self._load_openface_features(engagement_df)
        # TODO: add more features if needed
        return InterPersenSEMPIDataset(openface_features, engagements, person_ids)

# %%
def create_dataloaders(dataset: InterPersenSEMPIDataset, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Create the dataloaders for the training and validation sets."""
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# %%
if __name__ == "__main__":
    ds_loader = DataSetLoader(DATA_PATH)
    dataset = ds_loader.get_dataset()

    train_loader, val_loader = create_dataloaders(dataset)
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size: {len(val_loader.dataset)}")

    print(dataset.get_metadata(0))

    for i, data in enumerate(train_loader):
        print(f"Batch {i}")
        if i == 2:
            break
        print(data['features'].shape)
        print(data['pids'])
        print(data['score'])

    # save the dataset and dataloaders with pickle
    import pickle

    with open(os.path.join(DATA_PATH, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)

class ListenerSpeakerFeatureDataset(Dataset):
    def __init__(self, csv_path, frame_length=64, root_dir=""):
        self.data = pd.read_csv(csv_path)
        self.frame_length = frame_length
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def load_audio_feature_pt(self, path):
        full_path = os.path.join(self.root_dir, path)
        try:
            tensor = torch.load(full_path)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            return tensor.T if tensor.ndim == 2 else tensor
        except Exception as e:
            print(f"Error loading audio feature from {full_path}:\n{e}")
            raise

    def load_video_feature_csv(self, path, exclude_cols=META_DATA_COLUMNS):
        full_path = os.path.join(self.root_dir, path)
    
        # print(f"Loading video feature from {full_path}")
        df = pd.read_csv(full_path)
        df = df.loc[:, ~df.columns.isin(exclude_cols)]
        return torch.tensor(df.values, dtype=torch.float32)

    def _pad_or_crop(self, feature_tensor):
        n_frames, n_features = feature_tensor.shape
        output = torch.zeros((self.frame_length, n_features))
        if n_frames >= self.frame_length:
            output[:] = feature_tensor[:self.frame_length]
        else:
            output[:n_frames] = feature_tensor
        return output.T  # shape: (n_features, frame_length)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        listener_audio = self.load_audio_feature_pt(row["listener_audio_path"])
        listener_video = self.load_video_feature_csv(row["listener_video_path"])
        speaker_audio = self.load_audio_feature_pt(row["speaker_audio_path"])
        speaker_video = self.load_video_feature_csv(row["speaker_video_path"])
        # print(listener_video.shape, speaker_video.shape)
        # print(listener_audio.shape, speaker_audio.shape)

        # listener_feat = self._pad_or_crop(listener_audio) + self._pad_or_crop(listener_video)
        # speaker_feat = self._pad_or_crop(speaker_audio) + self._pad_or_crop(speaker_video)

        listener_feat = self._pad_or_crop(listener_video)
        speaker_feat = self._pad_or_crop(speaker_video)

        # features = torch.stack([listener_feat, speaker_feat], dim=0)  # shape: (2, n_features, frame_length)
        features = (speaker_feat,listener_feat)
        engagement = torch.tensor(float(row["engagement"]), dtype=torch.float32)
        pids = torch.tensor([0, 1], dtype=torch.int64)

        return {
            "features": features,
            "score": engagement,
            "pids": pids
        }


