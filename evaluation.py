import torch
import logging
from typing import Dict, List, Any
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handles evaluation of FrankenModels on specific tasks."""

    def __init__(self, model, tokenizer, device: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def prepare_dataset(self, task_name: str, split: str = "validation"):
        """Prepare dataset for evaluation based on task."""
        if task_name == "text_classification":
            # Example using IMDb dataset for sentiment classification, should change
            dataset = load_dataset("imdb", split=split)
            return self._prepare_classification_data(dataset)
        else:
            raise ValueError(f"Unsupported task: {task_name}")

    def _prepare_classification_data(self, dataset) -> DataLoader:
        """Prepare classification dataset."""

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], padding="max_length", truncation=True, max_length=512
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        return DataLoader(tokenized_dataset, batch_size=8, shuffle=False)

    def evaluate_classification(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on classification task."""
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                }

                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["label"].cpu().numpy())

        return {
            "accuracy": accuracy_score(all_labels, all_predictions),
            "f1": f1_score(all_labels, all_predictions, average="weighted"),
        }

    def evaluate_task(self, task_name: str) -> Dict[str, float]:
        """Main evaluation function."""
        logger.info(f"Starting evaluation for task: {task_name}")

        try:
            # Prepare dataset
            dataloader = self.prepare_dataset(task_name)

            # Evaluate based on task
            if task_name == "text_classification":
                results = self.evaluate_classification(dataloader)
            else:
                raise ValueError(f"Unsupported task: {task_name}")

            logger.info(f"Evaluation results: {results}")
            return results

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise


def compare_models(
    original_results: Dict[str, float], franken_results: Dict[str, float]
) -> Dict[str, float]:
    """Compare performance between original and FrankenModel."""
    comparison = {}
    for metric in original_results:
        difference = franken_results[metric] - original_results[metric]
        relative_change = (difference / original_results[metric]) * 100
        comparison[f"{metric}_difference"] = difference
        comparison[f"{metric}_relative_change"] = relative_change

    return comparison
