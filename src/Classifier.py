from tensorflow import keras # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras import layers # type: ignore
import tensorflow as tf # type: ignore
import os
import matplotlib.pyplot as plt # type: ignore
from PIL import Image # type: ignore

class Classifier(object):
    def __init__(self,name):
        self.name = name
        self.model = None

    def loadModel(self):
        # Carica il modello dal file .keras
        self.model = keras.models.load_model(self.name)
    
    def doPrediction(self,img_path):
        img_normalized = self.preprocess_image(image_path=img_path)

        # Fai la previsione
        prediction = self.model.predict(img_normalized)
        # print(prediction) => es. [[0.9950331]]

        # Interpretazione
        print(f"Probabilità classe positiva: {prediction[0][0]:.3f} e {1 - prediction[:,0]}")
        return prediction[0][0]

    def doPredictions(self, folder_path):
        numFalsi = 0
        # Scorri tutti i file della cartella
        for filename in sorted(os.listdir(folder_path)): # itera su tutti i nomi dei file contenuti nella cartella specificata
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')): # verifica che sia un'immagine
                img_path = os.path.join(folder_path, filename) # si ricava il path completo dell'immagine
                print(f"\nImmagine: {filename}")
                if self.doPrediction(img_path) <= 0.5:
                    numFalsi = numFalsi +1
        
        print("Totale falsi",numFalsi)
        return numFalsi
    
    def preprocess_image(self, image_path):
        img_size = (512, 512)
        # Carica l'immagine
        image = Image.open(image_path)
        
        # Converte in grayscale se necessario (L in PIL vuol dire scala dei grigi)
        if image.mode != 'L':
            image = image.convert('L')
        
        image = image.resize(img_size)
        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0 # tensorflow lavora con float32
        
        # Aggiunge le dimensioni per batch e canale
        # Da (512, 512) a (1, 512, 512, 1)
        # In che senso? Il modello è stato addestrato su input con forma: (batch_size, height, width, channels)
        # Se do una singola immagine (512, 512), il modello si aspetta anche la dim del batch che in questo caso è 1
        # e la dim del canale
        img_array = np.expand_dims(img_array, axis=0)  # batch dimension
        img_array = np.expand_dims(img_array, axis=-1)  # channel dimension
            
        return img_array
    
    def show_prediction_result(self, image_path, result):
        # Carica e mostra l'immagine originale
        image = Image.open(image_path)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(image, cmap='gray' if image.mode == 'L' else None)
        plt.title(f"Predizione: {result}", fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

name = "modello.keras"

if __name__ == "__main__":
    c = Classifier(name)
    c.loadModel()

    # Predizioni singole
    #c.doPrediction("./dataset/1_real/real_unknown_0100.png")
    c.doPrediction("./dataset/0_generated/generated_00400.png")

    # Predizioni su cartelle di immagini
    #c.doPredictions("./dataset/1_real")
    #c.doPredictions("./dataset/0_generated")