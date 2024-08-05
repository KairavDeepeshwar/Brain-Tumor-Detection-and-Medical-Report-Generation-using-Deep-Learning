# Brain Tumor Detection and Classification Using Deep Learning

## Project Overview

This project aims to develop deep learning models for the detection and classification of brain tumors using MRI images. The project utilizes multiple architectures, including VGG16, ResNet, EfficientNet, and ResNet50, to evaluate their performance in identifying various types of brain tumors. Additionally, a YOLOv5 model is trained on a brain tumor dataset from Roboflow for object detection. The Gemini API from Google Cloud is integrated to generate medical reports based on the detected tumor details. This project contributes to advancements in medical imaging and diagnostics and then utilize ReportLab to create detailed PDF medical reports, which include diagnostic findings, tumor classifications, and visualizations from the model outputs. This feature ensures that the results are easily accessible and presentable for clinical use.

### Key Components

1. **Data Preprocessing:** Loading and preprocessing MRI images, including resizing, normalization, and data augmentation.

2. **Model Architectures:**
   - **VGG16:** Pre-trained model fine-tuned on the custom dataset.
   - **ResNet:** Leveraging residual learning for deeper networks.
   - **EfficientNet:** Optimizing accuracy and computational resources.
   - **ResNet50:** A variant of ResNet for performance comparison.
   - **YOLOv5:** Object detection model trained on a brain tumor dataset from Roboflow.

3. **Training Process:** Compiling and fitting each model to the training data while monitoring performance on a validation set.

4. **Model Evaluation:** Evaluating each model's performance using accuracy, precision, recall, and F1 score, along with confusion matrices and classification reports.

5. **Visualization:** Visualizing training and validation metrics to analyze model performance.

6. **Gemini API Integration:** Generating medical reports based on the detected tumor details using the Gemini API from Google Cloud.
   
7. **Medical Report Generation:** Use ReportLab to create PDF reports with diagnostic findings, tumor classifications, and visualizations, making results accessible for clinical use.

### Test Accuracies
- **CNN:** Test Accuracy: 92.5%
- **VGG16:** Test Accuracy: 92.12%
- **ResNet50:** Test Accuracy: 90.81%
- **EfficientNet:** Test Accuracy: 75.24%
- **ResNet:** Test Accuracy: 74.28%

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pandas
- OpenCV
- Google Cloud SDK

## Getting Started

1. Clone the repository to your local machine.
2. Install the required packages using `pip install -r requirements.txt`.
3. Set up your Google Cloud credentials and project ID.
4. Open the notebooks in Jupyter Notebook or Google Colab.
5. Run the cells sequentially to execute the models and generate medical reports.

## Conclusion

This project demonstrates the application of deep learning techniques for medical image analysis, specifically for brain tumor detection. The results obtained from the various models and the integration of the Gemini API can serve as a foundation for further research and development in this field.

## License

This project is licensed under the MIT License.
