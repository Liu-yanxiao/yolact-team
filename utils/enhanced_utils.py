"""Enhanced utilities for YOLACT: data processing, metrics, and misc features"""

import os
import json
import pickle
import numpy as np
import torch
import cv2
from collections import defaultdict, OrderedDict
from pathlib import Path
import time
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import csv

class DatasetAnalyzer:
    """Analyzes dataset statistics and provides insights"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.class_counts = defaultdict(int)
        self.bbox_sizes = defaultdict(list)
        self.aspect_ratios = []
        self.image_sizes = []
        self.annotations_per_image = []
        self.total_images = 0
        
    def add_annotation(self, class_id: int, bbox: Tuple[float, float, float, float], 
                      image_size: Tuple[int, int]):
        """Add annotation for analysis"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        self.class_counts[class_id] += 1
        self.bbox_sizes[class_id].append((width, height))
        
        if height > 0:
            self.aspect_ratios.append(width / height)
    
    def add_image(self, image_size: Tuple[int, int], num_annotations: int):
        """Add image information"""
        self.image_sizes.append(image_size)
        self.annotations_per_image.append(num_annotations)
        self.total_images += 1
    
    def get_statistics(self) -> Dict:
        """Get comprehensive dataset statistics"""
        stats = {
            'total_images': self.total_images,
            'total_annotations': sum(self.class_counts.values()),
            'class_distribution': dict(self.class_counts),
            'avg_annotations_per_image': np.mean(self.annotations_per_image) if self.annotations_per_image else 0,
            'median_annotations_per_image': np.median(self.annotations_per_image) if self.annotations_per_image else 0,
            'avg_aspect_ratio': np.mean(self.aspect_ratios) if self.aspect_ratios else 0,
            'median_aspect_ratio': np.median(self.aspect_ratios) if self.aspect_ratios else 0,
        }
        
        avg_bbox_sizes = {}
        for class_id, sizes in self.bbox_sizes.items():
            widths = [s[0] for s in sizes]
            heights = [s[1] for s in sizes]
            avg_bbox_sizes[class_id] = {
                'avg_width': np.mean(widths),
                'avg_height': np.mean(heights),
                'median_width': np.median(widths),
                'median_height': np.median(heights),
            }
        stats['avg_bbox_sizes'] = avg_bbox_sizes
        
        return stats
    
    def save_report(self, output_path: str):
        """Save analysis report to file"""
        stats = self.get_statistics()
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Dataset analysis saved to {output_path}")


class BatchProcessor:
    """Utility for processing data in batches"""
    
    def __init__(self, batch_size: int = 32, shuffle: bool = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def process(self, data: List, process_fn, show_progress: bool = True):
        """Process data in batches with a processing function"""
        if self.shuffle:
            indices = np.random.permutation(len(data))
            data = [data[i] for i in indices]
        
        results = []
        total_batches = (len(data) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_result = process_fn(batch)
            results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
            
            if show_progress:
                batch_num = i // self.batch_size + 1
                print(f"Processed batch {batch_num}/{total_batches}", end='\r')
        
        if show_progress:
            print()  # New line after progress
        
        return results


class ImagePreprocessor:
    """Advanced image preprocessing utilities"""
    
    @staticmethod
    def normalize_image(image: np.ndarray, mean: Tuple[float, float, float], 
                       std: Tuple[float, float, float]) -> np.ndarray:
        """Normalize image with mean and std"""
        image = image.astype(np.float32)
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        return image
    
    @staticmethod
    def resize_with_aspect_ratio(image: np.ndarray, target_size: int, 
                                 max_size: int = None) -> Tuple[np.ndarray, float]:
        """Resize image maintaining aspect ratio"""
        h, w = image.shape[:2]
        scale = target_size / min(h, w)
        
        if max_size is not None:
            scale = min(scale, max_size / max(h, w))
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, scale
    
    @staticmethod
    def pad_to_size(image: np.ndarray, target_size: Tuple[int, int], 
                    pad_value: int = 0) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Pad image to target size"""
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        
        if pad_h > 0 or pad_w > 0:
            if len(image.shape) == 3:
                padded = np.full((target_h, target_w, image.shape[2]), pad_value, dtype=image.dtype)
                padded[:h, :w, :] = image
            else:
                padded = np.full((target_h, target_w), pad_value, dtype=image.dtype)
                padded[:h, :w] = image
            return padded, (pad_h, pad_w)
        
        return image, (0, 0)

class MetricsCalculator:
    """Enhanced metrics calculation for model evaluation"""
    
    def __init__(self, num_classes: int, iou_thresholds: List[float] = None):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.true_positives = defaultdict(lambda: defaultdict(int))
        self.false_positives = defaultdict(lambda: defaultdict(int))
        self.false_negatives = defaultdict(lambda: defaultdict(int))
        self.predictions_per_class = defaultdict(int)
        self.ground_truths_per_class = defaultdict(int)
    
    def update(self, pred_boxes: np.ndarray, pred_classes: np.ndarray, pred_scores: np.ndarray,
               gt_boxes: np.ndarray, gt_classes: np.ndarray):
        """Update metrics with predictions and ground truths"""
        for threshold in self.iou_thresholds:
            matched_gt = set()
            sorted_indices = np.argsort(-pred_scores)
            
            for idx in sorted_indices:
                pred_box = pred_boxes[idx]
                pred_class = pred_classes[idx]
                self.predictions_per_class[pred_class] += 1
                
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
                    if gt_class != pred_class or gt_idx in matched_gt:
                        continue
                    
                    iou = self._calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= threshold and best_gt_idx != -1:
                    self.true_positives[threshold][pred_class] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    self.false_positives[threshold][pred_class] += 1
            
            for gt_idx, gt_class in enumerate(gt_classes):
                self.ground_truths_per_class[gt_class] += 1
                if gt_idx not in matched_gt:
                    self.false_negatives[threshold][gt_class] += 1
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    
    def compute_ap(self, threshold: float, class_id: int) -> float:
        """Compute Average Precision for a class at given IoU threshold"""
        tp = self.true_positives[threshold].get(class_id, 0)
        fp = self.false_positives[threshold].get(class_id, 0)
        fn = self.false_negatives[threshold].get(class_id, 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return precision * recall
    
    def compute_map(self, threshold: float = 0.5) -> float:
        """Compute mean Average Precision across all classes"""
        aps = []
        for class_id in range(self.num_classes):
            ap = self.compute_ap(threshold, class_id)
            aps.append(ap)
        return np.mean(aps) if aps else 0.0
    
    def get_precision_recall(self, threshold: float = 0.5, class_id: int = None) -> Tuple[float, float]:
        """Get precision and recall for a specific class or overall"""
        if class_id is not None:
            tp = self.true_positives[threshold].get(class_id, 0)
            fp = self.false_positives[threshold].get(class_id, 0)
            fn = self.false_negatives[threshold].get(class_id, 0)
        else:
            tp = sum(self.true_positives[threshold].values())
            fp = sum(self.false_positives[threshold].values())
            fn = sum(self.false_negatives[threshold].values())
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return precision, recall


class ConfusionMatrix:
    """Confusion matrix for multi-class detection"""
    
    def __init__(self, num_classes: int, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    
    def update(self, pred_classes: np.ndarray, gt_classes: np.ndarray):
        """Update confusion matrix"""
        for pred, gt in zip(pred_classes, gt_classes):
            self.matrix[gt, pred] += 1
    
    def get_matrix(self) -> np.ndarray:
        """Get the confusion matrix"""
        return self.matrix
    
    def plot(self, save_path: str = None, normalize: bool = False):
        """Plot confusion matrix"""
        matrix = self.matrix.copy()
        
        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = matrix.astype(np.float32) / (row_sums + 1e-10)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(matrix, cmap='Blues')
        
        ax.set_xticks(np.arange(len(self.class_names)))
        ax.set_yticks(np.arange(len(self.class_names)))
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.set_yticklabels(self.class_names)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Ground Truth')
        ax.set_title('Confusion Matrix')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class PerformanceProfiler:
    """Profile model performance (speed, memory, etc.)"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset profiling data"""
        self.forward_times = []
        self.preprocessing_times = []
        self.postprocessing_times = []
        self.total_times = []
        self.memory_usage = []
    
    def add_timing(self, forward_time: float, preprocess_time: float = 0, 
                   postprocess_time: float = 0):
        """Add timing information"""
        self.forward_times.append(forward_time)
        self.preprocessing_times.append(preprocess_time)
        self.postprocessing_times.append(postprocess_time)
        self.total_times.append(forward_time + preprocess_time + postprocess_time)
    
    def add_memory(self, memory_mb: float):
        """Add memory usage in MB"""
        self.memory_usage.append(memory_mb)
    
    def get_statistics(self) -> Dict:
        """Get profiling statistics"""
        stats = {
            'avg_forward_time': np.mean(self.forward_times) if self.forward_times else 0,
            'median_forward_time': np.median(self.forward_times) if self.forward_times else 0,
            'avg_total_time': np.mean(self.total_times) if self.total_times else 0,
            'fps': 1.0 / np.mean(self.total_times) if self.total_times and np.mean(self.total_times) > 0 else 0,
            'avg_memory_mb': np.mean(self.memory_usage) if self.memory_usage else 0,
            'peak_memory_mb': max(self.memory_usage) if self.memory_usage else 0,
        }
        return stats
    
    def print_summary(self):
        """Print performance summary"""
        stats = self.get_statistics()
        print("\n" + "="*50)
        print("Performance Profiling Summary")
        print("="*50)
        print(f"Average Forward Time: {stats['avg_forward_time']*1000:.2f} ms")
        print(f"Median Forward Time: {stats['median_forward_time']*1000:.2f} ms")
        print(f"Average Total Time: {stats['avg_total_time']*1000:.2f} ms")
        print(f"FPS: {stats['fps']:.2f}")
        print(f"Average Memory: {stats['avg_memory_mb']:.2f} MB")
        print(f"Peak Memory: {stats['peak_memory_mb']:.2f} MB")
        print("="*50 + "\n")


# ========================== MISCELLANEOUS UTILITIES ==========================

class CheckpointManager:
    """Manage model checkpoints with versioning and cleanup"""
    
    def __init__(self, save_dir: str, max_keep: int = 5, keep_interval: int = 10000):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.keep_interval = keep_interval
        self.checkpoints = []
    
    def save_checkpoint(self, state_dict: Dict, iteration: int, prefix: str = "model"):
        """Save checkpoint and manage cleanup"""
        filename = f"{prefix}_iter_{iteration}.pth"
        filepath = self.save_dir / filename
        
        torch.save(state_dict, filepath)
        self.checkpoints.append((iteration, filepath))
        print(f"Checkpoint saved: {filepath}")
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond max_keep"""
        if len(self.checkpoints) <= self.max_keep:
            return
        
        self.checkpoints.sort(key=lambda x: x[0])
        to_keep = []
        to_remove = []
        
        for iteration, filepath in self.checkpoints:
            if iteration % self.keep_interval == 0 or len(self.checkpoints) - len(to_remove) <= self.max_keep:
                to_keep.append((iteration, filepath))
            else:
                to_remove.append((iteration, filepath))
        
        for iteration, filepath in to_remove:
            if filepath.exists():
                filepath.unlink()
                print(f"Removed old checkpoint: {filepath}")
        
        self.checkpoints = to_keep
    
    def load_latest_checkpoint(self) -> Optional[Dict]:
        """Load the latest checkpoint"""
        if not self.checkpoints:
            return None
        
        latest_iteration, latest_path = max(self.checkpoints, key=lambda x: x[0])
        if latest_path.exists():
            return torch.load(latest_path)
        return None


class ResultExporter:
    """Export results in various formats"""
    
    @staticmethod
    def export_detections_json(detections: List[Dict], output_path: str):
        """Export detections to JSON format"""
        with open(output_path, 'w') as f:
            json.dump(detections, f, indent=2)
        print(f"Detections exported to {output_path}")
    
    @staticmethod
    def export_detections_csv(detections: List[Dict], output_path: str):
        """Export detections to CSV format"""
        if not detections:
            print("No detections to export")
            return
        
        fieldnames = ['image_id', 'class_id', 'class_name', 'score', 'bbox_x1', 
                     'bbox_y1', 'bbox_x2', 'bbox_y2']
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for det in detections:
                row = {
                    'image_id': det.get('image_id', ''),
                    'class_id': det.get('class_id', ''),
                    'class_name': det.get('class_name', ''),
                    'score': det.get('score', 0),
                    'bbox_x1': det.get('bbox', [0, 0, 0, 0])[0],
                    'bbox_y1': det.get('bbox', [0, 0, 0, 0])[1],
                    'bbox_x2': det.get('bbox', [0, 0, 0, 0])[2],
                    'bbox_y2': det.get('bbox', [0, 0, 0, 0])[3],
                }
                writer.writerow(row)
        
        print(f"Detections exported to {output_path}")
    
    @staticmethod
    def export_metrics_report(metrics: Dict, output_path: str):
        """Export metrics report to text file"""
        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Model Evaluation Metrics Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            for key, value in metrics.items():
                if isinstance(value, dict):
                    f.write(f"\n{key}:\n")
                    for sub_key, sub_value in value.items():
                        f.write(f"  {sub_key}: {sub_value}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print(f"Metrics report exported to {output_path}")


class VisualizationHelper:
    """Helper for creating visualizations"""
    
    @staticmethod
    def plot_training_curves(train_losses: List[float], val_losses: List[float] = None,
                            save_path: str = None):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        
        if val_losses:
            plt.plot(val_losses, label='Validation Loss', linewidth=2)
        
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_class_distribution(class_counts: Dict[int, int], class_names: List[str],
                               save_path: str = None):
        """Plot class distribution histogram"""
        classes = sorted(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        labels = [class_names[c] if c < len(class_names) else f"Class {c}" for c in classes]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(classes)), counts)
        plt.xticks(range(len(classes)), labels, rotation=45, ha='right')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Class distribution plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_pr_curve(precisions: List[float], recalls: List[float], 
                     save_path: str = None, class_name: str = ""):
        """Plot Precision-Recall curve"""
        plt.figure(figsize=(8, 8))
        plt.plot(recalls, precisions, linewidth=2, marker='o', markersize=4)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve {class_name}')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"PR curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class DataAugmentationHelper:
    """Additional data augmentation utilities"""
    
    @staticmethod
    def random_brightness(image: np.ndarray, delta: int = 32) -> np.ndarray:
        """Randomly adjust brightness"""
        if np.random.rand() < 0.5:
            delta = np.random.uniform(-delta, delta)
            image = image.astype(np.float32)
            image += delta
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image
    
    @staticmethod
    def random_contrast(image: np.ndarray, lower: float = 0.5, upper: float = 1.5) -> np.ndarray:
        """Randomly adjust contrast"""
        if np.random.rand() < 0.5:
            alpha = np.random.uniform(lower, upper)
            image = image.astype(np.float32)
            image *= alpha
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image
    
    @staticmethod
    def random_saturation(image: np.ndarray, lower: float = 0.5, upper: float = 1.5) -> np.ndarray:
        """Randomly adjust saturation"""
        if np.random.rand() < 0.5:
            alpha = np.random.uniform(lower, upper)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = hsv[:, :, 1].astype(np.float32) * alpha
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255).astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return image
    
    @staticmethod
    def random_hue(image: np.ndarray, delta: int = 18) -> np.ndarray:
        """Randomly adjust hue"""
        if np.random.rand() < 0.5:
            delta = np.random.uniform(-delta, delta)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(np.float32) + delta) % 180
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return image


class ModelEvaluator:
    """Comprehensive model evaluation utilities"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate_on_batch(self, images: torch.Tensor, targets: List[Dict]) -> Dict:
        """Evaluate model on a batch of images"""
        with torch.no_grad():
            images = images.to(self.device)
            predictions = self.model(images)
        
        batch_metrics = {
            'num_predictions': len(predictions),
            'avg_confidence': 0,
            'num_targets': len(targets),
        }
        
        return batch_metrics
    
    def calculate_inference_time(self, image_size: Tuple[int, int], num_runs: int = 100) -> Dict:
        """Calculate average inference time"""
        dummy_input = torch.randn(1, 3, *image_size).to(self.device)
        
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = self.model(dummy_input)
                torch.cuda.synchronize() if self.device == 'cuda' else None
                times.append(time.time() - start)
        
        return {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'fps': 1.0 / np.mean(times),
        }


class ResultComparator:
    """Compare results from different models or configurations"""
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, name: str, metrics: Dict):
        """Add result from a model/config"""
        self.results[name] = metrics
    
    def compare(self, metric_name: str) -> Dict:
        """Compare a specific metric across all results"""
        comparison = {}
        for name, metrics in self.results.items():
            if metric_name in metrics:
                comparison[name] = metrics[metric_name]
        return comparison
    
    def get_best(self, metric_name: str, higher_is_better: bool = True) -> Tuple[str, float]:
        """Get the best result for a metric"""
        comparison = self.compare(metric_name)
        if not comparison:
            return None, None
        
        if higher_is_better:
            best_name = max(comparison.keys(), key=lambda k: comparison[k])
        else:
            best_name = min(comparison.keys(), key=lambda k: comparison[k])
        
        return best_name, comparison[best_name]
    
    def export_comparison(self, output_path: str):
        """Export comparison table"""
        if not self.results:
            print("No results to compare")
            return
        
        all_metrics = set()
        for metrics in self.results.values():
            all_metrics.update(metrics.keys())
        
        with open(output_path, 'w') as f:
            f.write("Model Comparison Report\n")
            f.write("="*80 + "\n\n")
            
            for metric in sorted(all_metrics):
                f.write(f"\n{metric}:\n")
                f.write("-"*40 + "\n")
                
                for name, metrics in self.results.items():
                    value = metrics.get(metric, 'N/A')
                    f.write(f"  {name:30s}: {value}\n")
        
        print(f"Comparison report exported to {output_path}")


# ========================== UTILITY FUNCTIONS ==========================

def create_output_directory(base_dir: str, experiment_name: str = None) -> Path:
    """Create timestamped output directory"""
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(base_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_experiment_config(config: Dict, output_dir: Path):
    """Save experiment configuration"""
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_path}")


def load_class_names(file_path: str) -> List[str]:
    """Load class names from file"""
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f]
    return class_names


def calculate_model_size(model: torch.nn.Module) -> Dict:
    """Calculate model size and parameter count"""
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = (param_count * 4) / (1024 ** 2)
    
    return {
        'total_parameters': param_count,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': param_count - trainable_params,
        'model_size_mb': model_size_mb,
    }


def setup_random_seed(seed: int = 42):
    """Setup random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_gpu_memory_info() -> Dict:
    """Get GPU memory information"""
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        
        info[f'GPU_{i}'] = {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - allocated,
        }
    
    return info


class DetectionAnalyzer:
    """Analyze detection results and patterns"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset analysis data"""
        self.detections_per_class = defaultdict(list)
        self.confidence_scores = defaultdict(list)
        self.bbox_areas = defaultdict(list)
        self.detection_counts_per_image = []
    
    def add_detections(self, image_id: str, detections: List[Dict]):
        """Add detections for an image"""
        self.detection_counts_per_image.append(len(detections))
        
        for det in detections:
            class_id = det.get('class_id', det.get('category_id'))
            score = det.get('score', det.get('confidence', 1.0))
            bbox = det.get('bbox')
            
            self.detections_per_class[class_id].append(image_id)
            self.confidence_scores[class_id].append(score)
            
            if bbox is not None:
                if len(bbox) == 4:
                    # x, y, w, h or x1, y1, x2, y2
                    if bbox[2] < bbox[0]:  # width format
                        area = bbox[2] * bbox[3]
                    else:  # x1,y1,x2,y2 format
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    self.bbox_areas[class_id].append(area)
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        stats = {
            'total_detections': sum(len(dets) for dets in self.detections_per_class.values()),
            'avg_detections_per_image': np.mean(self.detection_counts_per_image) if self.detection_counts_per_image else 0,
            'detections_per_class': {k: len(v) for k, v in self.detections_per_class.items()},
            'avg_confidence_per_class': {},
            'avg_bbox_area_per_class': {},
        }
        
        for class_id, scores in self.confidence_scores.items():
            stats['avg_confidence_per_class'][class_id] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'median': np.median(scores),
                'min': np.min(scores),
                'max': np.max(scores),
            }
        
        for class_id, areas in self.bbox_areas.items():
            stats['avg_bbox_area_per_class'][class_id] = {
                'mean': np.mean(areas),
                'std': np.std(areas),
                'median': np.median(areas),
            }
        
        return stats
    
    def get_low_confidence_detections(self, threshold: float = 0.5) -> Dict:
        """Get statistics on low confidence detections"""
        low_conf_stats = {}
        for class_id, scores in self.confidence_scores.items():
            low_conf = [s for s in scores if s < threshold]
            low_conf_stats[class_id] = {
                'count': len(low_conf),
                'percentage': len(low_conf) / len(scores) * 100 if scores else 0,
            }
        return low_conf_stats


class HeatmapGenerator:
    """Generate detection heatmaps for spatial analysis"""
    
    def __init__(self, image_size: Tuple[int, int], grid_size: int = 50):
        self.image_size = image_size
        self.grid_size = grid_size
        self.heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.detection_count = 0
    
    def add_detection(self, bbox: Tuple[float, float, float, float], weight: float = 1.0):
        """Add a detection to the heatmap"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        grid_x = int(cx / self.image_size[1] * self.grid_size)
        grid_y = int(cy / self.image_size[0] * self.grid_size)
        
        grid_x = max(0, min(self.grid_size - 1, grid_x))
        grid_y = max(0, min(self.grid_size - 1, grid_y))
        
        self.heatmap[grid_y, grid_x] += weight
        self.detection_count += 1
    
    def get_heatmap(self, normalize: bool = True) -> np.ndarray:
        """Get the heatmap"""
        if normalize and self.detection_count > 0:
            return self.heatmap / np.max(self.heatmap)
        return self.heatmap
    
    def plot(self, save_path: str = None, title: str = "Detection Heatmap"):
        """Plot the heatmap"""
        plt.figure(figsize=(10, 8))
        plt.imshow(self.get_heatmap(normalize=True), cmap='hot', interpolation='bilinear')
        plt.colorbar(label='Detection Density')
        plt.title(title)
        plt.xlabel('X Grid')
        plt.ylabel('Y Grid')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class AnnotationConverter:
    """Convert between different annotation formats"""
    
    @staticmethod
    def coco_to_voc(bbox: List[float], image_size: Tuple[int, int] = None) -> List[float]:
        """Convert COCO format (x, y, w, h) to VOC format (xmin, ymin, xmax, ymax)"""
        x, y, w, h = bbox
        return [x, y, x + w, y + h]
    
    @staticmethod
    def voc_to_coco(bbox: List[float]) -> List[float]:
        """Convert VOC format (xmin, ymin, xmax, ymax) to COCO format (x, y, w, h)"""
        xmin, ymin, xmax, ymax = bbox
        return [xmin, ymin, xmax - xmin, ymax - ymin]
    
    @staticmethod
    def normalize_bbox(bbox: List[float], image_size: Tuple[int, int]) -> List[float]:
        """Normalize bounding box coordinates to [0, 1]"""
        height, width = image_size
        if len(bbox) == 4:
            return [
                bbox[0] / width,
                bbox[1] / height,
                bbox[2] / width,
                bbox[3] / height,
            ]
        return bbox
    
    @staticmethod
    def denormalize_bbox(bbox: List[float], image_size: Tuple[int, int]) -> List[float]:
        """Denormalize bounding box coordinates from [0, 1] to pixel values"""
        height, width = image_size
        if len(bbox) == 4:
            return [
                bbox[0] * width,
                bbox[1] * height,
                bbox[2] * width,
                bbox[3] * height,
            ]
        return bbox


class DataSplitter:
    """Split dataset into train/val/test sets"""
    
    @staticmethod
    def split_data(data: List, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                   test_ratio: float = 0.15, shuffle: bool = True, seed: int = None) -> Dict:
        """Split data into train/val/test sets"""
        if seed is not None:
            np.random.seed(seed)
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
        
        n = len(data)
        indices = np.arange(n)
        
        if shuffle:
            np.random.shuffle(indices)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        return {
            'train': [data[i] for i in train_indices],
            'val': [data[i] for i in val_indices],
            'test': [data[i] for i in test_indices],
            'train_indices': train_indices.tolist(),
            'val_indices': val_indices.tolist(),
            'test_indices': test_indices.tolist(),
        }
    
    @staticmethod
    def stratified_split(data: List, labels: List[int], train_ratio: float = 0.7,
                        val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = None) -> Dict:
        """Stratified split maintaining class distribution"""
        if seed is not None:
            np.random.seed(seed)
        
        assert len(data) == len(labels), "Data and labels must have same length"
        
        class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for class_id, indices in class_indices.items():
            np.random.shuffle(indices)
            n = len(indices)
            
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            train_indices.extend(indices[:train_end])
            val_indices.extend(indices[train_end:val_end])
            test_indices.extend(indices[val_end:])
        
        return {
            'train': [data[i] for i in train_indices],
            'val': [data[i] for i in val_indices],
            'test': [data[i] for i in test_indices],
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
        }


class LossTracker:
    """Track and analyze training losses"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.losses = defaultdict(list)
        self.iterations = []
    
    def add_loss(self, iteration: int, **losses):
        """Add loss values for an iteration"""
        self.iterations.append(iteration)
        for name, value in losses.items():
            self.losses[name].append(value)
    
    def get_moving_average(self, loss_name: str, window_size: int = None) -> List[float]:
        """Get moving average of a loss"""
        if window_size is None:
            window_size = self.window_size
        
        losses = self.losses.get(loss_name, [])
        if not losses:
            return []
        
        moving_avg = []
        for i in range(len(losses)):
            start = max(0, i - window_size + 1)
            moving_avg.append(np.mean(losses[start:i+1]))
        
        return moving_avg
    
    def get_latest_losses(self, n: int = 10) -> Dict:
        """Get the latest n loss values"""
        latest = {}
        for name, values in self.losses.items():
            latest[name] = values[-n:] if len(values) >= n else values
        return latest
    
    def plot_losses(self, save_path: str = None, smooth: bool = True):
        """Plot all tracked losses"""
        plt.figure(figsize=(12, 6))
        
        for name, values in self.losses.items():
            if smooth:
                values = self.get_moving_average(name)
            plt.plot(self.iterations[:len(values)], values, label=name, linewidth=2)
        
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Loss plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_to_csv(self, output_path: str):
        """Export losses to CSV"""
        with open(output_path, 'w', newline='') as f:
            loss_names = list(self.losses.keys())
            writer = csv.writer(f)
            writer.writerow(['iteration'] + loss_names)
            
            for i, iteration in enumerate(self.iterations):
                row = [iteration]
                for name in loss_names:
                    row.append(self.losses[name][i] if i < len(self.losses[name]) else '')
                writer.writerow(row)
        
        print(f"Losses exported to {output_path}")


class ImageTiler:
    """Tile large images for processing"""
    
    def __init__(self, tile_size: int = 512, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = overlap
    
    def tile_image(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """Split image into tiles with positions"""
        h, w = image.shape[:2]
        tiles = []
        
        stride = self.tile_size - self.overlap
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                
                tile = image[y:y_end, x:x_end]
                
                if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
                    if len(tile.shape) == 3:
                        padded = np.zeros((self.tile_size, self.tile_size, tile.shape[2]), dtype=tile.dtype)
                        padded[:tile.shape[0], :tile.shape[1], :] = tile
                    else:
                        padded = np.zeros((self.tile_size, self.tile_size), dtype=tile.dtype)
                        padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded
                
                tiles.append((tile, (y, x)))
        
        return tiles
    
    def merge_tiles(self, tiles: List[Tuple[np.ndarray, Tuple[int, int]]], 
                   original_size: Tuple[int, int]) -> np.ndarray:
        """Merge tiles back into full image"""
        h, w = original_size
        
        if len(tiles) > 0 and len(tiles[0][0].shape) == 3:
            merged = np.zeros((h, w, tiles[0][0].shape[2]), dtype=tiles[0][0].dtype)
        else:
            merged = np.zeros((h, w), dtype=tiles[0][0].dtype)
        
        counts = np.zeros((h, w), dtype=np.float32)
        
        for tile, (y, x) in tiles:
            y_end = min(y + self.tile_size, h)
            x_end = min(x + self.tile_size, w)
            
            tile_h = y_end - y
            tile_w = x_end - x
            
            merged[y:y_end, x:x_end] += tile[:tile_h, :tile_w]
            counts[y:y_end, x:x_end] += 1
        
        counts[counts == 0] = 1
        if len(merged.shape) == 3:
            for i in range(merged.shape[2]):
                merged[:, :, i] /= counts
        else:
            merged /= counts
        
        return merged.astype(tiles[0][0].dtype)


if __name__ == '__main__':
    print("Enhanced utilities for YOLACT project")
    print("Available classes:")
    print("  - DatasetAnalyzer: Analyze dataset statistics")
    print("  - BatchProcessor: Process data in batches")
    print("  - ImagePreprocessor: Image preprocessing utilities")
    print("  - MetricsCalculator: Calculate evaluation metrics")
    print("  - ConfusionMatrix: Generate confusion matrices")
    print("  - PerformanceProfiler: Profile model performance")
    print("  - CheckpointManager: Manage model checkpoints")
    print("  - ResultExporter: Export results in various formats")
    print("  - VisualizationHelper: Create visualizations")
    print("  - DataAugmentationHelper: Additional augmentation utilities")
    print("  - ModelEvaluator: Comprehensive model evaluation")
    print("  - ResultComparator: Compare multiple model results")
    print("  - DetectionAnalyzer: Analyze detection patterns")
    print("  - HeatmapGenerator: Generate detection heatmaps")
    print("  - AnnotationConverter: Convert annotation formats")
    print("  - DataSplitter: Split datasets strategically")
    print("  - LossTracker: Track and visualize training losses")
    print("  - ImageTiler: Tile large images for processing")

