import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import os
import argparse
from utils import clean_str, remove_punctuation

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tqdm.pandas()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Sentence Transformer Model
model = SentenceTransformer("/home/models/paraphrase-multilingual-mpnet-base-v2", device=device)

def exact_match_binary(y_true, y_pred):
    """Computes exact match accuracy."""
    y_pred = str(y_pred)
    
    if isinstance(y_true, list):
        acc_abs_list, acc_prob_list = [], []
        for x in y_true:
            x = clean_text(x)
            if x in y_pred:
                return 1, 1
            x_tokens, y_tokens = set(x.split()), set(y_pred.split())
            acc_prob = len(x_tokens & y_tokens) / len(x_tokens)
            acc_abs = int(acc_prob == 1)
            acc_abs_list.append(acc_abs)
            acc_prob_list.append(acc_prob)

        return max(acc_abs_list), max(acc_prob_list)

    return (1, 1) if y_true in y_pred else (int(len(set(y_true.split()) & set(y_pred.split())) / len(y_true.split()) == 1), len(set(y_true.split()) & set(y_pred.split())) / len(y_true.split()))

def clean_text(text):
    """Cleans text input, removing punctuation and handling JSON parsing."""
    try:
        answer = eval(text)
        if "ANSWER" not in answer:
            raise ValueError(f"Ground truth might be absent: {text}")
        text = answer["ANSWER"]
    except:
        pass
    
    text = str(text) if isinstance(text, (str, float)) else "<Empty>"
    return remove_punctuation(text.lower().strip().replace("-", ""))

def semantic_match(y_true, y_pred):
    """Computes semantic similarity score using embeddings."""
    embeddings1, embeddings2 = model.encode(y_true), model.encode([str(y_pred)])
    return float(model.similarity(embeddings1, embeddings2).max())

def compute_accuracy(df, prediction_col, reference_col, similarity_threshold=0.95):
    """Computes accuracy metrics for the given DataFrame."""
    if prediction_col not in df or reference_col not in df:
        raise KeyError("Prediction or reference column missing.")

    df[reference_col] = df[reference_col].progress_apply(lambda x: [s.lower() for s in x])
    df["prediction_cleaned"] = df[prediction_col].progress_apply(clean_text)

    df["accuracy_exact_match"] = df.progress_apply(
        lambda row: exact_match_binary(row[reference_col], row["prediction_cleaned"])[0], axis=1
    )

    if "predictions_translated" in df.columns:
        df["predictions_translated"] = df["predictions_translated"].progress_apply(clean_text)
        df["accuracy_exact_match_(y_pred_translated)"] = df.progress_apply(
            lambda row: exact_match_binary(row[reference_col], row["predictions_translated"])[0], axis=1
        )
        df["accuracy_semantic_match_(y_pred_translated)"] = df.progress_apply(
            lambda row: semantic_match(row[reference_col], row["predictions_translated"])
            if row["accuracy_exact_match_(y_pred_translated)"] != 1 else 1,
            axis=1
        ).apply(lambda x: int(x >= similarity_threshold))

    accuracy_cols = [col for col in df.columns if "accuracy_" in col]
    df["accuracy"] = df[accuracy_cols].max(axis=1)

    return df

def main():
    parser = argparse.ArgumentParser(description="Compute accuracy metrics for predictions.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--savefile", type=str, required=True, help="Path to save the output CSV file.")
    parser.add_argument("--prediction_col", type=str, default="answer_en", help="Column containing predictions.")
    parser.add_argument("--reference_col", type=str, default="target", help="Column containing reference text.")
    parser.add_argument("--similarity_threshold", type=float, default=0.95, help="Semantic similarity threshold.")

    args = parser.parse_args()

    df_pred = pd.read_pickle(args.input_csv)
    df_pred = compute_accuracy(df_pred, args.prediction_col, args.reference_col, args.similarity_threshold)

    df_pred[["id", "aggregated_target", "prediction_cleaned", "accuracy_exact_match", "accuracy"]].to_csv(
        args.savefile, index=False
    )
    print(f"Accuracy computed and saved to: {args.savefile}")

if __name__ == "__main__":
    main()
