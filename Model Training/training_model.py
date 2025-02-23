from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch
import multiprocessing

if __name__ == '__main__':
    # Only needed if creating an executable with something like pyinstaller
    #multiprocessing.freeze_support()

    # Ensure output directory exists
    output_dir = "traffic_detection/yolov8m_finetune"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load YOLOv8m model (pre-trained on COCO)
    model = YOLO("yolov8m.pt")

    # 2. Fine-Tune without forgetting COCO knowledge
    model.train(
        data=r"C:/Users/garg1/OneDrive/Desktop/100K/dataset/data.yaml",
        epochs=8,  # Fine-tuning
        imgsz=640,  # Better for small objects
        batch=8,  # Adjust based on GPU memory
        device=0,  # Use GPU
        workers=4,
        project="traffic_detection",
        name="yolov8m_finetune",
        cache=False,
        amp=True,
        optimizer="AdamW",
        lr0=0.0005,  # Lower LR to avoid catastrophic forgetting
        lrf=0.1,
        cos_lr=True,
        freeze=10  # Freezing first 10 layers to retain pre-trained COCO features
    )

    print("✅ Fine-Tuning Completed!")

    # 3. Validate and extract results
    metrics = model.val()
    results = metrics.results_dict

    # Get class names from model
    class_names = model.names

    # Generate Training Metrics Plots
    try:
        plt.figure(figsize=(12, 8))

        metrics_names = ['metrics/precision', 'metrics/recall', 'metrics/mAP50', 'metrics/mAP50-95']

        for metric in metrics_names:
            if metric in results:
                plt.plot(results[metric], label=metric)

        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Metrics over Epochs')
        plt.legend()
        plt.grid(True)

        # Save and display the figure
        metrics_path = os.path.join(output_dir, 'training_metrics.png')
        plt.savefig(metrics_path)
        plt.show()
        plt.close()
    except Exception as e:
        print(f"⚠️ Error in training metric visualization: {e}")

    # Generate Confusion Matrix
    try:
        if hasattr(metrics, "confusion_matrix"):
            confusion_matrix = metrics.confusion_matrix.matrix.cpu().numpy()

            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

            cm_path = os.path.join(output_dir, 'confusion_matrix.png')
            plt.savefig(cm_path)
            plt.show()
            plt.close()
        else:
            print("❌ Confusion Matrix not available.")
    except Exception as e:
        print(f"⚠️ Error in confusion matrix visualization: {e}")

    print("✅ Training & Evaluation Completed!")