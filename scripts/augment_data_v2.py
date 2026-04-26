import pandas as pd
import os
import random

DATASET_PATH = os.path.join('data', 'dataset.csv')

new_data = [
    # --- QUESTIONS ---
    ("Is it good ?", "Positive"),
    ("Was the service bad ?", "Negative"),
    ("Is this the best phone ever ?", "Positive"),
    ("Why is this so terrible ?", "Negative"),
    ("Did it work correctly ?", "Positive"),
    ("Is this product worth the money ?", "Positive"),
    ("Can this get any worse ?", "Negative"),
    ("Are you serious with this quality ?", "Negative"),
    ("Is the battery life acceptable ?", "Positive"),
    ("Was the delivery on time ?", "Positive"),
    
    # --- MIXED SENTIMENT (Target Sentiment Dominates) ---
    ("The design is good but the battery is terrible.", "Negative"),
    ("I love the color but the size is totally wrong.", "Negative"),
    ("It looks cheap but works perfectly fine.", "Positive"),
    ("Shipping was slow but the product is amazing.", "Positive"),
    ("Great features but horrible customer support.", "Negative"),
    ("It has flaws but I absolutely love it.", "Positive"),
    ("The packaging was nice but the item arrived broken.", "Negative"),
    ("Usually I hate this brand but this model is fantastic.", "Positive"),
    ("Not bad looking but it breaks after one use.", "Negative"),
    ("Very expensive but totally worth the premium price.", "Positive"),

    # --- SARCASM / CONTRAST ---
    ("Great, another problem to deal with.", "Negative"),
    ("Oh brilliant, exactly what I did not want.", "Negative"),
    ("Wow, this lasted a whole two days. Impressive.", "Negative"),
    ("What a completely wonderful waste of time.", "Negative"),
    ("Yeah right, like this is actually going to work.", "Negative"),
    ("Fantastic job breaking my phone.", "Negative"),
    ("Sure, I love waiting on hold for three hours.", "Negative"),
    ("Best purchase ever if you want to throw away your money.", "Negative"),

    # --- COMPARATIVES ---
    ("This phone is better than the previous one.", "Positive"),
    ("Much worse than I expected compared to the photos.", "Negative"),
    ("Faster than the older model.", "Positive"),
    ("Not as good as the competition.", "Negative"),
    ("A massive upgrade from last year.", "Positive"),
    ("Way cheaper feeling than the original version.", "Negative"),
    ("Superior quality compared to everything else.", "Positive"),
    ("Less durable than my last pair.", "Negative"),

    # --- SHORT EXPRESSIONS ---
    ("Good.", "Positive"),
    ("Bad.", "Negative"),
    ("Not great.", "Negative"),
    ("Terrible.", "Negative"),
    ("Awesome.", "Positive"),
    ("Horrible.", "Negative"),
    ("Excellent.", "Positive"),
    ("Okay.", "Neutral"),
    ("Fine.", "Neutral"),
    ("Meh.", "Neutral"),
    ("Whatever.", "Neutral"),
    ("Standard.", "Neutral"),
    ("Average.", "Neutral"),
    ("Decent.", "Neutral"),
    ("Acceptable.", "Neutral"),

    # --- MORE NEUTRAL (Factual/Observations) ---
    ("It is what it is.", "Neutral"),
    ("The device has a 5 inch screen.", "Neutral"),
    ("It comes in three colors.", "Neutral"),
    ("The manual is included in the box.", "Neutral"),
    ("It requires two AA batteries.", "Neutral"),
    ("The package was left at the door.", "Neutral"),
    ("I bought this on a Tuesday.", "Neutral"),
    ("It arrived yesterday.", "Neutral"),
    ("The color is a standard black.", "Neutral"),
    ("It weighs exactly what the box says.", "Neutral"),
    ("The cord is three feet long.", "Neutral"),
    ("It is made of plastic.", "Neutral"),
    ("There are buttons on the side.", "Neutral"),
    ("The app has a menu.", "Neutral"),
    ("It is neither good nor bad, just average.", "Neutral"),
    ("Nothing special at all.", "Neutral"),
    ("It does the basic job.", "Neutral"),

    # --- MORE POSITIVE ---
    ("Absolutely fantastic product.", "Positive"),
    ("I highly recommend this to anyone.", "Positive"),
    ("Could not be happier with my purchase.", "Positive"),
    ("Five stars all the way.", "Positive"),
    ("Superb build quality and excellent performance.", "Positive"),
    ("It works flawlessly every time.", "Positive"),
    ("The best investment I have made this year.", "Positive"),
    ("Incredibly fast and reliable.", "Positive"),
    ("Beautiful design and very functional.", "Positive"),

    # --- MORE NEGATIVE ---
    ("Completely unacceptable quality.", "Negative"),
    ("Do not buy this garbage.", "Negative"),
    ("Worst customer service I have ever dealt with.", "Negative"),
    ("It stopped working after five minutes.", "Negative"),
    ("A complete rip-off and a scam.", "Negative"),
    ("Terrible experience overall.", "Negative"),
    ("I demand a full refund for this trash.", "Negative"),
    ("It is completely useless.", "Negative"),
    ("Extremely disappointed in this brand.", "Negative"),
]

# Multiply to generate ~600 entries to balance and boost dataset
augmented_set = []
for _ in range(5):
    augmented_set.extend(new_data)

df_new = pd.DataFrame(augmented_set, columns=['text', 'sentiment'])

if os.path.exists(DATASET_PATH):
    df_old = pd.read_csv(DATASET_PATH)
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    
    # Shuffle
    df_combined = df_combined.sample(frac=1).reset_index(drop=True)
    
    df_combined.to_csv(DATASET_PATH, index=False)
    
    print(f"Added {len(df_new)} diverse samples.")
    print(f"Total dataset size: {len(df_combined)}")
    print("Class Distribution:")
    print(df_combined['sentiment'].value_counts())
else:
    df_new.to_csv(DATASET_PATH, index=False)
    print("Created new dataset.")
