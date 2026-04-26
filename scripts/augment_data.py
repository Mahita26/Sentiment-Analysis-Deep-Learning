import pandas as pd
import os

DATASET_PATH = os.path.join('data', 'dataset.csv')

new_data = [
    # Punctuation & Questions
    ("is that nice ?", "Positive"),
    ("is that good ?", "Positive"),
    ("is that bad ?", "Negative"),
    ("that is nice", "Positive"),
    ("this is nice", "Positive"),
    ("that is not nice", "Negative"),
    ("this is not nice", "Negative"),
    ("is this nice ?", "Positive"),
    ("is this bad ?", "Negative"),
    ("that is very good !", "Positive"),
    ("is it okay ?", "Neutral"),
    ("is it fine ?", "Neutral"),
    
    # Context / Short
    ("not good", "Negative"),
    ("not bad", "Positive"),
    ("not nice", "Negative"),
    ("it is okay", "Neutral"),
    ("it is bad", "Negative"),
    ("it is nice", "Positive"),
]

df_new = pd.DataFrame(new_data, columns=['text', 'sentiment'])

if os.path.exists(DATASET_PATH):
    df_old = pd.read_csv(DATASET_PATH)
    if "is that nice ?" not in df_old['text'].values:
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_csv(DATASET_PATH, index=False)
        print(f"Added {len(df_new)} new samples. Total samples: {len(df_combined)}")
    else:
        print("Data already augmented.")
else:
    df_new.to_csv(DATASET_PATH, index=False)
    print("Created new dataset.")
