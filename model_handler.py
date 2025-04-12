import torch
import logging
from typing import List, Dict, Any
from transformers import PreTrainedModel
from copy import deepcopy

logger = logging.getLogger(__name__)


class LayerDuplicator:
    """Handles the duplication of transformer layers in LLaMA models."""

    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.original_config = deepcopy(model.config)

    def get_layer_mapping(
        self, layers_to_duplicate: List[int], num_duplications: int
    ) -> Dict[int, int]:
        """Create a mapping of original layers to their duplicated positions."""
        mapping = {}
        current_position = 0

        for layer_idx in range(self.model.config.num_hidden_layers):
            if layer_idx in layers_to_duplicate:
                # Map original layer and its duplicates
                for dup in range(num_duplications + 1):  # +1 for original layer
                    mapping[current_position] = layer_idx
                    current_position += 1
            else:
                # Map non-duplicated layer
                mapping[current_position] = layer_idx
                current_position += 1

        return mapping

    def duplicate_layers(
        self, layers_to_duplicate: List[int], num_duplications: int
    ) -> PreTrainedModel:
        """Duplicate specified layers in the model."""
        logger.info(
            f"Duplicating layers {layers_to_duplicate} {num_duplications} times"
        )

        # Validate input
        max_layer = max(layers_to_duplicate)
        if max_layer >= self.model.config.num_hidden_layers:
            raise ValueError("Layer index exceeds model size")

        # Create new model with expanded architecture
        new_num_layers = (
            self.model.config.num_hidden_layers
            + len(layers_to_duplicate) * num_duplications
        )

        # Get layer mapping
        layer_mapping = self.get_layer_mapping(layers_to_duplicate, num_duplications)

        # Create new layers list
        new_layers = []
        for new_idx in range(len(layer_mapping)):
            original_idx = layer_mapping[new_idx]
            new_layers.append(deepcopy(self.model.model.layers[original_idx]))

        # Update model architecture
        self.model.config.num_hidden_layers = new_num_layers
        self.model.model.layers = torch.nn.ModuleList(new_layers)

        return self.model

    def verify_duplication(self) -> bool:
        """Verify that layer duplication was successful."""
        # Basic verification of model structure
        try:
            # Perform a simple forward pass
            dummy_input = torch.zeros((1, 10), dtype=torch.long).to(self.model.device)
            _ = self.model(dummy_input)
            return True
        except Exception as e:
            logger.error(f"Layer duplication verification failed: {str(e)}")
            return False


def create_franken_model(
    base_model: PreTrainedModel, layers_to_duplicate: List[int], num_duplications: int
) -> PreTrainedModel:
    """Create a FrankenModel by duplicating specified layers."""
    duplicator = LayerDuplicator(base_model)
    modified_model = duplicator.duplicate_layers(layers_to_duplicate, num_duplications)

    if duplicator.verify_duplication():
        logger.info("FrankenModel created successfully")
        return modified_model
    else:
        raise RuntimeError("Failed to create FrankenModel")
