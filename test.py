import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageTk

class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Emotion Recognition")
        
        self.model = load_model('model_file.h5')
        
        self.root.geometry("1400x900")
        self.root.configure(bg='#333')
        
        self.content_frame = tk.Frame(self.root, bg='#333')
        self.content_frame.pack(expand=True, fill=tk.BOTH)
        
        self.left_frame = tk.Frame(self.content_frame, bg='#333', width=700, highlightbackground="white", highlightthickness=2)
        self.left_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        self.img_label = tk.Label(self.left_frame, text="Image Expression Recognition", font=("Arial", 20), bg='#333', fg='white')
        self.img_label.pack(pady=10)
        
        self.browse_button = tk.Button(self.left_frame, text="Browse Image", command=self.browse_file, font=("Arial", 15))
        self.browse_button.pack(pady=10)
        
        self.image_label = tk.Label(self.left_frame, bg='black')
        self.image_label.pack(pady=10)
        
        self.emotion_label = tk.Label(self.left_frame, text="Detected Emotion:", font=("Arial", 15), bg='#333', fg='white')
        self.emotion_label.pack(pady=10)
        
        self.emotion_result_label = tk.Label(self.left_frame, text="", font=("Arial", 15), bg='#333', fg='white')
        self.emotion_result_label.pack(pady=10)
        
        # Create a button to open the progress bars window
        self.open_progress_window_button = tk.Button(self.left_frame, text="Show Progress Bars", command=self.open_progress_window, font=("Arial", 15))
        self.open_progress_window_button.pack(pady=10)
        
        self.right_frame = tk.Frame(self.content_frame, bg='#333', width=700, highlightbackground="white", highlightthickness=2)
        self.right_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        self.vid_label = tk.Label(self.right_frame, text="Real-Time Video Recognition", font=("Arial", 20), bg='#333', fg='white')
        self.vid_label.pack(pady=10)
        
        self.video_label = tk.Label(self.right_frame, bg='black')
        self.video_label.pack(pady=10)
        
        self.start_stop_button = tk.Button(self.right_frame, text="Start Real-Time Recognition", command=self.toggle_realtime_recognition, font=("Arial", 15))
        self.start_stop_button.pack(pady=10)
        
        self.video_capture = cv2.VideoCapture(0)
        self.face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
        self.recognizing = False
        
        self.pause_video()
        
        self.progress_window = None
        
        # Define selected_image_path attribute
        self.selected_image_path = None
        
        # Add exit button
        self.exit_button = tk.Button(self.root, text='Quit', fg="red", command=self.root.destroy, font=('arial', 25, 'bold'))
        self.exit_button.pack(side=tk.BOTTOM)

    def browse_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.selected_image_path = file_path
            self.display_image_and_emotion()

    def display_image_and_emotion(self):
        if self.selected_image_path:
            img = cv2.imread(self.selected_image_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_detect.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                sub_face_img = gray_img[y:y+h, x:x+w]
                resized = cv2.resize(sub_face_img, (48, 48))
                normalize = resized / 255.0
                reshaped = np.reshape(normalize, (1, 48, 48, 1))
                result = self.model.predict(reshaped)
                
                # Update progress bars based on the prediction
                self.update_progress_bars(result[0])
                
                label = np.argmax(result, axis=1)[0]
                
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.rectangle(img, (x, y), (x+w, y+h), (50, 50, 255), 2)
                cv2.rectangle(img, (x, y-40), (x+w, y), (50, 50, 255), -1)
                cv2.putText(img, self.labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.image_label.configure(image=imgtk)
            self.image_label.image = imgtk  
            
            self.emotion_result_label.configure(text=f"Detected Emotion: {self.labels_dict[label]}")

    def update_progress_bars(self, emotion_data):
        if self.progress_window:
            # Update progress bars in the separate window
            for i, emotion in enumerate(self.progress_labels):
                value = emotion_data[i] * 100
                if emotion in self.progress_vars:
                    self.progress_vars[emotion]['value'] = value
                    self.progress_labels_widgets[emotion].config(text=f"{emotion}: {int(value)}%")

    def update_video(self):
        ret, frame = self.video_capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                sub_face_img = gray[y:y+h, x:x+w]
                resized = cv2.resize(sub_face_img, (48, 48))
                normalize = resized / 255.0
                reshaped = np.reshape(normalize, (1, 48, 48, 1))
                result = self.model.predict(reshaped)
                label = np.argmax(result, axis=1)[0]
                
                # Update progress bars based on the prediction
                self.update_progress_bars(result[0])
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
                cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
                cv2.putText(frame, self.labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame_tk = ImageTk.PhotoImage(image=frame)
            
            self.video_label.configure(image=frame_tk)
            self.video_label.image = frame_tk  
            
            if self.recognizing:
                self.root.after(10, self.update_video)
        else:
            self.video_capture.release()
            cv2.destroyAllWindows()
            self.pause_video()

    def toggle_realtime_recognition(self):
        if self.recognizing:
            self.pause_video()
        else:
            self.start_video_recognition()

    def start_video_recognition(self):
        self.recognizing = True
        self.start_stop_button.config(text="Stop Real-Time Recognition")
        self.update_video()

    def pause_video(self):
        self.recognizing = False
        self.start_stop_button.config(text="Start Real-Time Recognition")

    def open_progress_window(self):
        if self.progress_window is None or not self.progress_window.winfo_exists():
            self.progress_window = tk.Toplevel(self.root)
            self.progress_window.title("Emotion Progress Bars")
            self.progress_window.geometry("400x400")
            self.progress_window.configure(bg='#333')

            # Create and pack progress bars
            self.progress_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            self.progress_vars = {}
            self.progress_labels_widgets = {}

            for emotion in self.progress_labels:
                label = tk.Label(self.progress_window, text=emotion, font=("Arial", 12), bg='#333', fg='white')
                label.pack(pady=2)
                progress = ttk.Progressbar(self.progress_window, orient="horizontal", length=300, mode="determinate", maximum=100)
                progress.pack(pady=2)
                self.progress_vars[emotion] = progress
                self.progress_labels_widgets[emotion] = label

if __name__ == '__main__':
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()
