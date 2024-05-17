import torch 
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        
        IoU_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        IoU_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        IoUs = torch.cat([IoU_b1.unsqueeze(0), IoU_b2.unsqueeze(0)], dim=0)
        IoU_max, bestbox = torch.max(IoUs, dim=0)
        identity_function = target[..., 20].unsqueeze(3)
        
        # Box Coordinates
        box_predictions = identity_function * (bestbox * predictions[..., 26:30] + 
                                               (1 - bestbox) * predictions[..., 21:25])
        box_targets = identity_function * target[..., 21:25]
        
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2), torch.flatten(box_targets, end_dim=-2))
        
        # Object Loss
        predicted_box = bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        object_loss = self.mse(torch.flatten(identity_function * predicted_box), 
                               torch.flatten(identity_function * target[..., 20:21]))
        
        # No Object Loss
        no_obj_loss = self.mse(
            torch.flatten((1 - identity_function) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - identity_function) * target[..., 20:21], start_dim=1)
        )
        no_obj_loss += self.mse(
            torch.flatten((1 - identity_function) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - identity_function) * target[..., 20:21], start_dim=1)
        )
        
        # Classification Loss
        classification_loss = self.mse(
            torch.flatten(identity_function * predictions[..., :20], end_dim=-2),
            torch.flatten(identity_function * target[..., :20], end_dim=-2)
        )
        
        # Total Loss
        total_loss = (
            self.lambda_coord * box_loss +
            object_loss +
            self.lambda_noobj * no_obj_loss +
            classification_loss
        )
        
        return total_loss
