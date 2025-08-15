import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def closest_point(position, points):
    min_distance = float("inf")
    closest_point = None
    point_index = None
    distance = float("inf")
    for i, point in enumerate(points):
        distance = (position - point).pow(2).mean()
        if distance < min_distance:
            closest_point = point
            min_distance = distance
            point_index = i
    return closest_point, distance, point_index


def unbatched_mapped_loss(prediction : Tensor, true_points : Tensor, distance_threshold = 0.02):
    for i, point in enumerate(true_points):
        if point.isnan().any():
            true_points = true_points[:i]
    points = prediction[:, :2]
    confidences = prediction[:,2]
    device = prediction.device
    distance_loss = torch.zeros((1,), device = prediction.device)
    classification_loss = torch.zeros((1,), device = prediction.device)
    correct_predictions = 0
    for true_point in true_points: # for each true_point move closest predicted point closer
        closest_prediction, distance, point_index = closest_point(true_point, points)
        correct_prediction = distance < distance_threshold
        correct_predictions += 1 if correct_prediction else 0
        distance_loss += (closest_prediction - true_point).pow(2).mean()
        classification_loss += -F.sigmoid(confidences[point_index]).log() if correct_prediction else 0
    
    classification_loss += -(1-F.sigmoid(confidences)).log().mean()*0.1 # decrease all confidenses
    

    """ if true_points.shape[0] > 0:
        for predicted_point in points: # for each predicted point move closer to the closest true point
            closest_true_point, distance, point_index = closest_point(predicted_point, true_points)
            distance_loss += (closest_true_point - predicted_point).pow(2).mean() * 0.5 """
    
    distance_loss /= true_points.shape[0] + points.shape[0]
    

    if "nan" in str(distance_loss.item()):
        print(f"nan in loss. {points.shape} {true_points.shape}")
    return distance_loss, classification_loss, correct_predictions

def mapped_loss(prediction_batched, true_points_batched, distance_threshold = 0.02):
    distance_loss, classification_loss, correct_predictions = 0, 0, 0
    for prediction, true_points in zip(prediction_batched, true_points_batched):
        d_l, c_l, correct = unbatched_mapped_loss(prediction, true_points, distance_threshold = distance_threshold)
        distance_loss += d_l
        classification_loss += c_l
        correct_predictions += correct
    distance_loss /= prediction_batched.size(0)
    classification_loss /= prediction_batched.size(0)
    return distance_loss, classification_loss, correct_predictions


class PointMatchingLoss(nn.Module):
    def __init__(self, position_weight = 1.0, classification_weight = 1.0, classification_distance = 0.01, classification_loss = nn.BCEWithLogitsLoss(), squared_distance=True):
        super().__init__()
        self.squared_distance = squared_distance
        self.position_weight = position_weight
        self.classification_weight = classification_weight
        self.classification_distance = classification_distance
        self.classification_loss = classification_loss
        

    def forward(self, prediction, true_points):
        """
        prediction: (B, N, 3)
        true_points: (B, M, 2)
        NaNs in either tensor indicate padded points (ignored in loss)
        """
        predicted_points = prediction[...,:2]
        predicted_logits = prediction[...,2]
        B, N, _ = predicted_points.shape
        _, M, _ = true_points.shape
        total_loss = 0.0
        count = 0

        n_correct_predictions = 0
        for b in range(B):
            # Mask out NaNs (padding)
            pred_mask = ~torch.isnan(predicted_points[b, :, 0])
            true_mask = ~torch.isnan(true_points[b, :, 0])

            pred = predicted_points[b, pred_mask]
            logits = predicted_logits[b, pred_mask]
            true = true_points[b, true_mask]

            if pred.size(0) == 0 or true.size(0) == 0:
                continue  # skip empty sets

            # Compute cost matrix
            diff = pred[:, None, :] - true[None, :, :]
            if self.squared_distance:
                cost_matrix = (diff ** 2).sum(dim=2).detach().cpu().numpy()
            else:
                cost_matrix = torch.norm(diff, dim=2).detach().cpu().numpy()

            # Hungarian matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_pred = pred[row_ind]
            matched_true = true[col_ind]

            # Loss for this batch item
            if self.squared_distance:
                distances = ((matched_pred - matched_true) ** 2).mean(-1)
                position_loss = torch.mean(distances)
                classes = torch.where(distances < self.classification_distance**2, torch.tensor(1.0), torch.tensor(0.0))
                
                n_correct_predictions += classes.sum().item()
            else:
                distances = torch.norm(matched_pred - matched_true, dim=1)
                position_loss = torch.mean(distances)
                classes = torch.where(distances < self.classification_distance, torch.tensor(1.0), torch.tensor(0.0))
                n_correct_predictions += classes.sum().item()

            targets = torch.zeros_like(logits) # everything exept closest close points should be classified as true

            targets[row_ind] = classes
            classification_loss = self.classification_loss(logits, targets)

            total_loss += position_loss * self.position_weight + classification_loss * self.classification_weight
            count += 1

        return total_loss / max(count, 1), n_correct_predictions

