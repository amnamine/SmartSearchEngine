"""Train two SmartSearchEngine models: boutique (dataset_train5) and catalogue (Clean_Catalogue)."""
from ai_test5 import SmartSearchEngine, DATASET_FILE, MODEL_FILE_STORE, MODEL_FILE_CATALOGUE

CATALOGUE_TRAIN = "dataset_catalogue_train.csv"


def main():
    print("=== 1/3 Building catalogue training CSV ===")
    from createdata_catalogue import create_catalogue_dataset
    create_catalogue_dataset()

    print("\n=== 2/3 Training boutique model ===")
    e1 = SmartSearchEngine()
    e1.train(DATASET_FILE, source_tag="boutique")
    e1.save_model(MODEL_FILE_STORE)

    print("\n=== 3/3 Training catalogue model ===")
    e2 = SmartSearchEngine()
    e2.train(CATALOGUE_TRAIN, source_tag="catalogue")
    e2.save_model(MODEL_FILE_CATALOGUE)

    print("\n[DONE] Saved:", MODEL_FILE_STORE, "and", MODEL_FILE_CATALOGUE)


if __name__ == "__main__":
    main()
