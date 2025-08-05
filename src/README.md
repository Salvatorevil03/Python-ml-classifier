## 📁 Project Structure

The project is divided into **four main Python files**, each handling a specific task:

- **`Trainer.py`**  
  Responsible for **creating and training** the classification model using TensorFlow. After training, the model is saved locally for future use.

- **`Classifier.py`**  
  Loads the pre-trained model and uses it to **perform predictions** on new MRI images, distinguishing between real and synthetic data.

- **`HeatMap.py`**  
  Generates **Grad-CAM heatmaps** to provide interpretability for the classifier's predictions. This helps identify the areas of the image that influenced the model's decision.

- **`Gui.py`**  
  Implements a simple **graphical user interface** using **Cluede Sonet**. The GUI allows users to interact with the classifier, visualize predictions, and view XAI heatmaps. It integrates functionalities from all the other modules.
