import torch
from typing import Tuple, List, Optional


SEGMENT_COLOR = {
    'ceiling' : [255, 255, 0],
    'wall' : [10, 255, 71], 
    'floor' : [255, 128, 128]
}


def convert_semantic_segmentation(
    class_queries: torch.Tensor, 
    mask_queries: torch.Tensor,
    target_sizes: Optional[Tuple[Tuple[int, int]]] = None
) -> List[torch.Tensor]:
    
    if not isinstance(class_queries, torch.Tensor):
        class_queries = torch.tensor(class_queries)
    if not isinstance(mask_queries, torch.Tensor):
        mask_queries = torch.tensor(mask_queries)
        
    # Remove the null class `[..., :-1]`
    masks_classes = class_queries.softmax(dim=-1)[..., :-1]
    masks_probs = mask_queries.sigmoid()  # [batch_size, num_queries, height, width]

    # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
    segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
    batch_size = mask_queries.shape[0]
    
    if target_sizes is not None:
        if batch_size != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )
        semantic_segmentation = []
        for idx in range(batch_size):
            resized_logits = torch.nn.functional.interpolate(
                segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
            )
            semantic_map = resized_logits[0].argmax(dim=0)
            semantic_segmentation.append(semantic_map)
    else:
        # If target_sizes is None, simply take the argmax across classes
        semantic_segmentation = segmentation.argmax(dim=1)
        semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

    return semantic_segmentation