import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

CSV_FILE = '/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/labels.csv'
captions = '/kaggle/input/memotion-with-captions/caption_BLIP.csv'
rationales = '/kaggle/input/memotion-rationales/rationales.csv'
seed = 42

def load_data(CSV_FILE, downsample=False, captions=None, rationales=None, seed=42):
    ''' Preprocess data with 3 columns: image name, text, and binary offensive label (0 not offensive 1 offensive) '''
    df = pd.read_csv(CSV_FILE)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df['offensive'] = np.where(df['offensive'] == 'not_offensive', 'not offensive', 'offensive')
    df['label'] = df['offensive'].map({'not offensive': 0, 'offensive': 1})
    df = df[['image_name', 'text_corrected', 'offensive', 'label']]
    df = df.rename(columns={'text_corrected': 'text'})
    df["offensive"] = df["offensive"].astype(str)
    df["text"] = df["text"].astype(str)
    df.loc[df['text'].isna(), 'text'] = 'nan'

    if captions is not None:
        df_captions = pd.read_csv(captions)
        df = pd.merge(df, df_captions, on='image_name', how='inner')
        df['combined'] = df.apply(lambda row: f"Given a Text: {row['text']}, which is embedded in an Image described by this caption: {row['caption']}. Predict whether the meme is offensive or not offensive.", axis=1)
        df['combined_distilled'] = df.apply(lambda row: f"Given a Text: {row['text']}, which is embedded in an Image described by this caption: {row['caption']}. please provide a streamlined rationale associated with the meme", axis=1)

    if rationales is not None:
        df_rationales = pd.read_csv(rationales)
        df = pd.merge(df, df_rationales, on='image_name', how='left')

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df['label'])
    df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=seed, stratify=df_train['label'])

    if downsample:
        df_nof = df_train[df_train['label'] == 0]
        df_of = df_train[df_train['label'] == 1]
        df_of_downsampled = resample(df_of, replace=False, n_samples=len(df_nof), random_state=seed)
        df_train = pd.concat([df_of_downsampled, df_nof])

    print('train : \n', df_train.label.value_counts())
    print('val : \n', df_val.label.value_counts())
    print('test : \n', df_test.label.value_counts())

    return df_train, df_val, df_test

def load_dataset_text_only(CSV_FILE, downsample=False, captions=None, rationales=None, mode='distil', seed=42) -> DatasetDict:
    """Load pandas dataset and split into train, validation, and test sets, then convert to Hugging Face Dataset."""

    df_train, df_val, df_test = load_data(CSV_FILE, downsample=downsample, captions=captions, rationales=rationales, seed=seed)
    
    if mode == 'distil':
        column_of_interest = ['combined_distilled', 'rationale']
    elif mode == 'label':
        column_of_interest = ['combined', 'offensive']
    elif mode == 'text_only':
        column_of_interest = ['text', 'offensive']

    df_train = df_train[column_of_interest]
    df_val = df_val[column_of_interest]
    df_test = df_test[column_of_interest]

    if mode == 'distil':
        df_train = df_train.rename(columns={'rationale': 'label', 'combined_distilled': 'text'})
        df_val = df_val.rename(columns={'rationale': 'label', 'combined_distilled': 'text'})
        df_test = df_test.rename(columns={'rationale': 'label', 'combined_distilled': 'text'})
    elif mode == 'label':
        df_train = df_train.rename(columns={'offensive': 'label', 'combined': 'text'})
        df_val = df_val.rename(columns={'offensive': 'label', 'combined': 'text'})
        df_test = df_test.rename(columns={'offensive': 'label', 'combined': 'text'})
    elif mode == 'text_only':
        df_train = df_train.rename(columns={'offensive': 'label'})
        df_val = df_val.rename(columns={'offensive': 'label'})
        df_test = df_test.rename(columns={'offensive': 'label'})

    # Ensuring 'text' and 'label' columns are strings
    for df_split in [df_train, df_val, df_test]:
        df_split["text"] = df_split["text"].astype(str)
        df_split["label"] = df_split["label"].astype(str)

    # Creating Hugging Face Datasets for each split
    dataset_train = Dataset.from_pandas(df_train[['text', 'label']])
    dataset_val = Dataset.from_pandas(df_val[['text', 'label']])
    dataset_test = Dataset.from_pandas(df_test[['text', 'label']])

    # Shuffling the datasets
    dataset_train = dataset_train.shuffle(seed=seed)
    dataset_val = dataset_val.shuffle(seed=seed)
    dataset_test = dataset_test.shuffle(seed=seed)

    # Combining the datasets into a DatasetDict
    dataset = DatasetDict({
        'train': dataset_train,
        'validation': dataset_val,
        'test': dataset_test
    })

    return dataset

# dataset_distil = load_dataset_text_only(CSV_FILE, downsample=False, captions=captions, rationales=rationales, mode='distil', seed=seed)
# dataset_label = load_dataset_text_only(CSV_FILE, downsample=False, captions=captions, rationales=rationales, mode='label', seed=seed)
# dataset_text_only = load_dataset_text_only(CSV_FILE, downsample=False, captions=captions, rationales=rationales, mode='text_only', seed=seed)

# print("Distil Mode - First Example:")
# print(dataset_distil['train'][0])
# print("\nLabel Mode - First Example:")
# print(dataset_label['train'][0])
# print("\nText Only Mode - First Example:")
# print(dataset_text_only['train'][0])