import tensorflow as tf # type: ignore
from tensorflow.keras import layers, models # type: ignore
import matplotlib.pyplot as plt # type: ignore
from tensorflow.keras import metrics # type: ignore

# con padding valid (ovvero 1)
# Output dim = (I - K + 1) / S
# I = dimensione di input
# K = dimensione filtro
# S = stride

class Trainer(object):
    def __init__(self,img_size,batch_size,pathDir):
        self.img_size = img_size
        self.batch_size = batch_size
        self.pathDir = pathDir
        self.train_ds = None
        self.val_ds = None
        self.history = None
        self.model = None
    
    def datasetLoad(self):
        # Caricamento dataset da cartella
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.pathDir,             # percorso cartella che contiene le due sotto cartelle
            validation_split=0.2,     # prende l'80% delle immagini del dataset come train, indicato in subset sotto
            subset="training",
            seed=123,                 # per riprodurre la casualità della divisione
            image_size=img_size,      # per cambiare se necessario la dimensione delle immagini
            batch_size=batch_size,
            color_mode='grayscale',   # B/N
            label_mode='binary'       # 0 = sintetiche, 1 = reali (dipende dall'ordine alfabetico delle dir, 0 chi viene prima)
        )
        # train_ds è un vettore di tuple del tipo (immagine, label)

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.pathDir,
            validation_split=0.2,
            subset="validation",      # indico che deve prendere il restante 20% per il validation set
            seed=123,
            image_size=img_size,
            batch_size=batch_size,
            color_mode='grayscale',
            label_mode='binary'
        )

        # Normalizzazione (da [0,255] a [0,1])
        normalization_layer = layers.Rescaling(1./255)
        self.train_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
        # con map, applico la func a ogni tupla di train_ds, in cui x è l'immagine e y la label (la quale resta invariata)
        self.val_ds = self.val_ds.map(lambda x, y: (normalization_layer(x), y))

        # Ottimizza le performance, caricando in RAM il dataset dopo la prima epoca
        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        # con shuffle, mischio ad ogni epoca l'ordine delle immagnini per evitare che il modello ne impari l'ordine
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    def modelCreation(self):
        # Modello CNN per immagini grayscale
        self.model = models.Sequential([ # Sequential per indicare che i layer sono sequenziali
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_size + (1,)),  # canale singolo di input (img B/W)
            # 32, (3, 3), activation='relu' => 32 filtri ognuno 3x3, post convoluzione passa tutto dentro ReLu
            # se non specificato il padding è valid (ovvero 1, ovvero non si aggiungono pixel ai bordi)
            # se non specificato la strides di default è di 1 pixel in orizzontale e verticale
            # parto con img 512 x 512, dopo la prima convoluzione ho 510 x 510 x 32
            layers.MaxPooling2D(),
            # se non si specifica, tensorflow usa MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')
            # ovvero, divide l'immagine in quadrati da 2x2, in cui si prende il max, il risultato e la divisione per 2 delle dim
            # ora l'img è 255 x 255 x 32
            layers.Conv2D(64, (3, 3), activation='relu'), # perchè il filtro è ancora 2D? perchè la depth del fitro viene
            # ricavata dalla depth dell'input, quindi qui ogni kernel è 3 x 3 x 32
            layers.MaxPooling2D(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(), # prende la feature map è la converte in un vettore 1D per poter essere ricevuto
            # dal layer dense, ad esempio, feature map da 64 x 64 x 128, lo converte in un vettore da 524288 = 64x64x128
            layers.Dense(128, activation='relu'), # layer da 128 neuroni, il cui output va tutto nel prossimo singolo neurone
            # essendo una dense, ogni singolo neurone riceve tutto il vettore 1D in ingresso
            # quindi ci sono 128 x dimVet1D pesi, + 128 di bias
            layers.Dense(1, activation='sigmoid')  # output binario
        ])
        # 512 x 512
        # 510 x 510 x 32
        # 255 x 255 x 32
        # 253 x 253 x 64
        # 126 x 126 x 64 (MaxPooling2D tronca per difetto)
        # 124 x 124 x 128
        # 62 x 62 x 128
        # 492032
        # 128
        # 1 (output)

        # Nota importante: conv2d_2 è il nome dato di default all'ultimo layer Conv2D (gli altri si chiamano conv2d, conv2d_1)

        self.model.compile(optimizer='adam',  # l'algortimo è sempre la backpropagation, ma adam è usa il valore del gradiente in modo più intelligente, varia dinamicamente anche il learning rate
              loss='binary_crossentropy',   # loss func
              # metriche da valutare sia nel train (fit) che nell'evaluate
              # area sottesa sotto la curva ROC
              metrics=[
                    'accuracy',
                    metrics.Precision(name='precision'),
                    metrics.Recall(name='recall'),
                    metrics.AUC(name='auc')
                ])

    def train(self):
        # Addestramento
        # history contiene a fine train, valori di loss e accuratezza, utili per realizzare grafici
        self.history = self.model.fit(self.train_ds, validation_data=self.val_ds, epochs=10)

    def evalue(self):
        # Valutazione
        results = self.model.evaluate(self.val_ds, return_dict=True)
        print(f"Risultati valutazione:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        self.showGraphic(history=self.history)

    def save(self):
        scelta = input("Vuoi salvare il modello? Y/y: ")
        if scelta == "y" or scelta == "Y":
            self.model.build(input_shape=(None, 512, 512, 1))
            self.model.save("modello.keras")
            print("Modello salvato con successo")
    
    def showGraphic(self,history):
        # Accuracy
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy durante il training')
        plt.xlabel('Epoche')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # Loss
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss durante il training')
        plt.xlabel('Epoche')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


# Parametri
img_size = (512, 512) # indico la grandezza che voglio per le img, potrei mettere anche quella reale
batch_size = 32
pathDir = "dataset"
    
if __name__ == "__main__":
    t = Trainer(img_size,batch_size,pathDir)
    t.datasetLoad()
    t.modelCreation()
    t.train()
    t.evalue()
    t.save()
