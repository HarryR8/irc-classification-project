"""
Prepare BUS-BRA dataset: load metadata, create patient-level splits.	

BUS-BRA structure:
- bus_data.csv: ID, Case, Histology, Pathology, BIRADS, Device, Width, Height, Side, BBOX
- Images: bus_XXXX-l.png, bus_XXXX-r.png (left/right views)
- Masks: mask_XXXX-l.png, mask_XXXX-r.png
- Case = patient ID (multiple images per patient)
 """

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

def create_patient_splits(
    data_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create train/val/test splits at PATIENT level (Case column).
    
    Critical: Multiple images per patient must stay together.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    df = pd.read_csv(data_dir / "bus_data.csv")
    print(f"Loaded {len(df)} images from {df['Case'].nunique()} patients")
    print(f"Pathology distribution:\n{df['Pathology'].value_counts()}")
    
    # Convert pathology to binary label
    df["label"] = (df["Pathology"] == "malignant").astype(int)
    
    # Verify each patient has a single label
    assert df.groupby("Case")["label"].nunique().max() == 1
    
    # Get unique patients with their label
    patient_labels = df.groupby("Case")["label"].first().reset_index()
    
    # Stratified patient split into test, then train/val - 2-way splitter (prevent leakage)
    
    # First split off test set (patient-level)
    train_val_cases, test_cases = train_test_split(
        patient_labels["Case"], # the array being split  
        test_size=test_ratio,   # put 15% of the items into the second output (test_cases)
        stratify=patient_labels["label"],   # when splitting patients, stratify by their label (preserve distribution in each split)
        random_state=seed,  # reproducible randomization
    )
    
    # Make stratification labels match the items being split in second split
    case_to_label = dict(zip(patient_labels["Case"], patient_labels["label"]))
    train_val_labels = [case_to_label[c] for c in train_val_cases]
    
    # Then split train/val (from remaining patients, only 75% of total)
    train_cases, val_cases = train_test_split(
        train_val_cases,
        test_size=val_ratio / (1 - test_ratio), # 10% of the total dataset, not 10% of train_val
        stratify=train_val_labels,  # use labels corresponding to train_val_cases
        random_state=seed,
    )
    
    # Assign splits
    # Convert arrays to sets for faster lookup (O(1) instead of O(n))
    train_cases, val_cases, test_cases = set(train_cases), set(val_cases), set(test_cases)
    
    def get_split(case):
        if case in test_cases:
            return "test"
        elif case in val_cases:
            return "val"
        return "train"
    
    df["split"] = df["Case"].apply(get_split)
    
    # Verify no leakage
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        c1 = set(df[df["split"] == s1]["Case"])
        c2 = set(df[df["split"] == s2]["Case"])
        assert len(c1 & c2) == 0, f"Leakage between {s1} and {s2}!"
    print("✓ No patient leakage")
    
    # Statistics
    print("\n--- Split Statistics ---")
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        n_img = len(split_df)
        n_pat = split_df["Case"].nunique()
        benign = (split_df["label"] == 0).sum()
        malignant = (split_df["label"] == 1).sum()
        print(f"{split:5s}: {n_img:4d} images, {n_pat:3d} patients (benign={benign}, malignant={malignant})")
    
    # Save
    cols = ["ID", "Case", "Pathology", "label", "BIRADS", "BBOX", "split"]
    df[cols].to_csv(output_dir / "splits.csv", index=False)
    print(f"\nSaved: {output_dir / 'splits.csv'}")
    
    # Save metadata
    meta = {
        "seed": seed,
        "n_images": len(df),
        "n_patients": df["Case"].nunique(),
        "splits": {s: len(df[df["split"] == s]) for s in ["train", "val", "test"]},
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    return df


def verify_images(df: pd.DataFrame, data_dir: Path):
    """Check images exist."""
    missing = []
    for img_id in df["ID"]:
        if not (data_dir / f"{img_id}.png").exists():
            missing.append(img_id)
    
    if missing:
        print(f"⚠ Missing {len(missing)} images")
    else:
        print(f"✓ All {len(df)} images found")


def main():
    data_dir = Path("data/raw")
    splits_dir = Path("data/splits")
    
    if not (data_dir / "bus_data.csv").exists():
        print(f"Error: {data_dir / 'bus_data.csv'} not found")
        print("Download BUS-BRA and place files in data/raw/")
        return
    
    df = create_patient_splits(data_dir, splits_dir)
    verify_images(df, data_dir)


if __name__ == "__main__":
    main()
