import tkinter as tk
from tkinter import messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD # type: ignore
from PIL import Image, ImageTk # type: ignore
from Classifier import Classifier
from HeatMap import HeatMap

class ImageDragDropGUI:
    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.root.title("Classificatore Immagini")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        self.image_path = None
        self.photo_image = None
        self.photo_image1 = None
        self.photo_image2 = None
        self.hm = HeatMap("modello.keras","conv2d_2")
        self.setup_ui()

    def setup_ui(self):
        # Titolo
        title_label = tk.Label(
            self.root,
            text="Trascina qui un'immagine per classificarla",
            font=('Arial', 18, 'bold'),
            bg='#f0f0f0',
            fg='#333333'
        )
        title_label.pack(pady=20)

        # Frame principale per contenere le due immagini
        self.main_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.main_frame.pack(pady=20, padx=20, expand=True, fill='both')

        # Frame sinistro per la prima immagine
        self.left_frame = tk.Frame(
            self.main_frame,
            width=350,
            height=300,
            bg='white',
            relief='solid',
            bd=2
        )
        self.left_frame.pack(side='left', padx=10, expand=True, fill='both')
        self.left_frame.pack_propagate(False)

        # Label per l'immagine sinistra
        self.left_label = tk.Label(
            self.left_frame,
            text="📁\n\nImmagine Originale\n\nTrascina qui un'immagine\n(JPG, PNG, BMP, GIF)",
            font=('Arial', 12),
            bg='white',
            fg='#666666',
            justify='center'
        )
        self.left_label.pack(expand=True)

        # Frame destro per la seconda immagine
        self.right_frame = tk.Frame(
            self.main_frame,
            width=350,
            height=300,
            bg='white',
            relief='solid',
            bd=2
        )
        self.right_frame.pack(side='right', padx=10, expand=True, fill='both')
        self.right_frame.pack_propagate(False)

        # Label per l'immagine destra
        self.right_label = tk.Label(
            self.right_frame,
            text="🔍\n\nImmagine per Analisi\n\nQui apparirà la mappa di calore\ndell'immagine da analizzare",
            font=('Arial', 12),
            bg='white',
            fg='#666666',
            justify='center'
        )
        self.right_label.pack(expand=True)

        # Frame per il risultato
        self.result_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.result_frame.pack(pady=20)

        # Label per il risultato
        self.result_label = tk.Label(
            self.result_frame,
            text="",
            font=('Arial', 14, 'bold'),
            bg='#f0f0f0',
            fg='#333333'
        )
        self.result_label.pack()

        # Configura drag and drop su entrambi i frame
        self.left_frame.drop_target_register(DND_FILES)
        self.left_frame.dnd_bind('<<Drop>>', self.on_drop)
        
        self.right_frame.drop_target_register(DND_FILES)
        self.right_frame.dnd_bind('<<Drop>>', self.on_drop)

    def on_drop(self, event):
        files = self.root.tk.splitlist(event.data)
        if len(files) > 0:
            file_path = files[0]
            if self.is_image_file(file_path):
                self.load_and_classify_image(file_path)
            else:
                messagebox.showerror("Errore", "File non valido. Usa JPG, PNG, BMP o GIF.")

    def is_image_file(self, file_path):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif')
        return file_path.lower().endswith(valid_extensions)

    def load_and_classify_image(self, file_path):
        try:
            self.image_path = file_path

            # 1
            # Carica l'immagine
            image1 = Image.open(file_path)
            
            # Ridimensiona l'immagine mantenendo le proporzioni
            image1.thumbnail((300, 200), Image.Resampling.LANCZOS)
            self.photo_image1 = ImageTk.PhotoImage(image1)

            # 2
            # Carica l'immagine
            self.hm.show_heatmap(file_path)
            image2 = Image.open("cam.jpg")
            
            # Ridimensiona l'immagine mantenendo le proporzioni
            image2.thumbnail((300, 200), Image.Resampling.LANCZOS)
            self.photo_image2 = ImageTk.PhotoImage(image2)

            # Mostra l'immagine in entrambi i label
            self.left_label.configure(image=self.photo_image1, text="")
            self.right_label.configure(image=self.photo_image2, text="")
            
            # Aggiorna il risultato per mostrare che sta processando
            self.result_label.configure(text="🔄 Analizzando l'immagine...", fg='#0066cc')
            
            # Forza l'aggiornamento dell'interfaccia
            self.root.update_idletasks()
            self.root.update()

            # Esegue la predizione
            name = "modello.keras"
            c = Classifier(name)
            c.loadModel()
            result = c.doPrediction(file_path)

            if result <= 0.5:
                # Mostra il risultato nell'interfaccia
                result_text = f"✅ Probabilità immagine sintetica: {(1-result):.3f}"
                if result > 0.5:
                    color = '#009900'  # Verde per risultato positivo
                else:
                    color = '#cc6600'  # Arancione per risultato negativo
            else:
                # Mostra il risultato nell'interfaccia
                result_text = f"✅ Probabilità immagine reale: {result:.3f}"
                if result > 0.5:
                    color = '#009900'  # Verde per risultato positivo
                else:
                    color = '#cc6600'  # Arancione per risultato negativo
                
            self.result_label.configure(text=result_text, fg=color)
            
            # Mostra anche il messagebox
            messagebox.showinfo("Risultato", f"Probabilità classe positiva: {result:.3f}")

        except Exception as e:
            self.result_label.configure(text="❌ Errore durante l'analisi", fg='#cc0000')
            messagebox.showerror("Errore", f"Errore: {str(e)}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    try:
        app = ImageDragDropGUI()
        app.run()
    except ImportError as e:
        print("Installa le dipendenze: pip install tkinterdnd2 Pillow")
        print(f"Errore: {e}")
    except Exception as e:
        print(f"Errore: {e}")