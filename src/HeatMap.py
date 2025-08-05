import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow import keras # type: ignore
from PIL import Image # type: ignore
from tensorflow.keras import Sequential, Input # type: ignore
# Display
from IPython.display import display # type: ignore
import IPython.display # type: ignore
import matplotlib as mpl # type: ignore
import matplotlib.pyplot as plt # type: ignore

class HeatMap(object):
    def __init__(self, name, layer_name):
        self.name = name
        self.layer_name = layer_name

        self.model1 = keras.models.load_model(self.name)

        # creo un nuovo modello a partire dal precedente
        # prendo come input layer, l'ultimo layer convolutivo e come output lascio lo stesso
        loaded_model = keras.models.load_model(self.name)
        n = 5 # posizione del layer dopo quello convolutivo che farà da input (parte da 0)
        input_tensor = loaded_model.layers[n].input # Prendo l'input del layer n-esimo
        # Ricostruisci il grafo dal layer n in poi
        x = input_tensor
        for layer in loaded_model.layers[n:]:
            x = layer(x)
        self.model2 = keras.models.Model(inputs=input_tensor, outputs=x)
        self.model2.layers[-1].activation = None


        # Creo 2 modelli per Grad-CAM (modelli nel senso di funziona computazionale con input e output)
        # in cui al primo come input gli do lo stesso input del modello originale e come output l'ultimo layer convolutivo
        # al secondo come input layer, do l'n-esimo layer e come output, la predizione
        self.grad_model1 = tf.keras.models.Model(
            inputs=self.model1.inputs,
            outputs=self.model1.get_layer(self.layer_name).output
        )

        self.grad_model2 = tf.keras.models.Model(
            inputs=self.model2.inputs,
            outputs=self.model2.outputs[0]
        )
    
    def make_gradcam_heatmap(self, img_array):
        # tensorflow registra le operazioni FATTE SUI TENSORI attravero il grafo computazionale e lo usa per il calcolo del gradiente
        # anche se gli passo un solo valore (tensore modo dimensionale), lui sa l'espressione utilzzata per calcolarlo
        # ovvero sa y = f(x)
        img_array = tf.cast(img_array, tf.float32)
        img_array = tf.Variable(img_array) # rendo l'input, una variabile per tensorflow, cosi può registare le operazioni su di essa
        with tf.GradientTape() as tape: # Apre un contesto che registra automaticamente tutte le operazioni sui tensori che sono all'interno del blocco with
            conv_outputs = self.grad_model1(img_array)
            predictions = self.grad_model2(conv_outputs)
            # Prendo la predizione, e vedo la classe scelta
            # in base alla classe scelta, calcolo l'espressione della sua probabilità, di cui farò il gradiente
            # predictions[:, 0] = seleziona tutti gli elementi del tensore (lista) e prendi il primo elemento
            # nel nostro caso predictions[:, 0] = [numero tra 0 e 1] (notare come sia ancora un tensore)
            result = predictions[0][0]
            result = tf.sigmoid(result)
            if result <= 0.5:
                output = 1 - predictions[:, 0] # operazione legale tra tensori
                # se la classe è negativa, ho bisogno di 1 - p e non mi basta solo p, perché col gradiente devo trovare i pixel delle feature map
                # che hanno contribuito maggiormente ad aumentare la probabilità della classe negativa
            else:
                output = predictions[:, 0]
            
            
        # Calcola i gradienti (tape è un oggetto che esiste anche al di fuori del contesto di with)
        grads = tape.gradient(predictions, conv_outputs)
        
        # Media globale dei gradienti (Global Average Pooling)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) # ora ho il peso di ogni gradiente
        # pooled_grads vettore di scalari
        
        conv_outputs = conv_outputs[0] # toglie la dim del batch dalle feature map (1, H, W, C) -> (H, W, C)
        # Moltiplica ogni canale delle feature maps per il corrispondente peso e li somma
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis] # ora ho una img dalla shape (H, W, 1)
        heatmap = tf.squeeze(heatmap) # rimuove le dimensioni di lunghezza 1 da un tensore, in questo caso il numero di canali
        
        # Passa il risulato attraverso una ReLu e normalizza
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        # Per poterla usare con matplotlib, cv2 o simili, la heatmap viene convertita da tensore a array NumPy.
        return heatmap.numpy()
    
    def show_heatmap(self,image_path):
        img_array = self.preprocess_image(image_path)
        heatmap = self.make_gradcam_heatmap(img_array)

        #plt.matshow(heatmap) # colora in automatico, valori bassi = blu scuro, valori medi = verde, valori alti = giallo
        #plt.show()
        self.save_and_display_gradcam(img_path=image_path, heatmap=heatmap)
    
    def save_and_display_gradcam(self,img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
        # Load the original image
        img = keras.utils.load_img(img_path)
        img = keras.utils.img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = mpl.colormaps["jet"]

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.utils.array_to_img(superimposed_img)

        # Save the superimposed image
        superimposed_img.save(cam_path)

        # Display Grad CAM
        display(IPython.display.Image(cam_path))
    
    def preprocess_image(self, image_path):
        img_size = (512, 512)
        image = Image.open(image_path)
        
        if image.mode != 'L':
            image = image.convert('L')
        
        image = image.resize(img_size)
        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0
        
        img_array = np.expand_dims(img_array, axis=0)  # batch dimension
        img_array = np.expand_dims(img_array, axis=-1)  # channel dimension
            
        return img_array

if __name__ == "__main__":
    hm = HeatMap("modello.keras","conv2d_2")
    hm.show_heatmap("./dataset/1_real/real_unknown_0100.png")