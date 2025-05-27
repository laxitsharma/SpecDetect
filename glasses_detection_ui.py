import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Try to import face module from opencv-contrib-python
try:
    import cv2.face
    FACE_AVAILABLE = True
except ImportError:
    FACE_AVAILABLE = False

class GlassesDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Glasses Detection Trainer")
        self.root.geometry("800x600")
        
        # Initialize detector
        self.detector = GlassesDetector()
        self.cap = None
        self.is_capturing = False
        self.current_mode = None  # 'with_glasses' or 'without_glasses'
        self.last_capture = 0
        self.capture_interval = 0.5  # seconds
        
        # Create main container
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create video frame
        self.video_frame = ttk.LabelFrame(self.main_frame, text="Camera Feed", padding="5")
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480, bg='black')
        self.canvas.pack()
        
        # Create control buttons frame
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=5)
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_label = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        # Buttons
        self.btn_frame = ttk.Frame(self.control_frame)
        self.btn_frame.pack(side=tk.LEFT, padx=5)
        
        self.btn_with_glasses = ttk.Button(self.btn_frame, text="Start With Glasses", 
                                         command=lambda: self.start_capture('with_glasses'))
        self.btn_with_glasses.pack(side=tk.LEFT, padx=2)
        
        self.btn_without_glasses = ttk.Button(self.btn_frame, text="Start Without Glasses",
                                            command=lambda: self.start_capture('without_glasses'))
        self.btn_without_glasses.pack(side=tk.LEFT, padx=2)
        
        self.btn_stop = ttk.Button(self.btn_frame, text="Stop", command=self.stop_capture, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=2)
        
        self.btn_train = ttk.Button(self.control_frame, text="Train Model", command=self.train_model)
        self.btn_train.pack(side=tk.LEFT, padx=5)
        
        self.btn_test = ttk.Button(self.control_frame, text="Test Detection", command=self.test_detection)
        self.btn_test.pack(side=tk.LEFT, padx=5)
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start video update
        self.update_video()
    
    def start_capture(self, mode):
        self.current_mode = mode
        self.is_capturing = True
        self.last_capture = time.time()
        self.status_var.set(f"Capturing: {'With' if mode == 'with_glasses' else 'Without'} Glasses")
        self.update_ui_state()
    
    def stop_capture(self):
        self.is_capturing = False
        self.current_mode = None
        self.status_var.set("Ready")
        self.update_ui_state()
    
    def update_ui_state(self):
        is_active = self.is_capturing
        self.btn_with_glasses.config(state=tk.DISABLED if is_active else tk.NORMAL)
        self.btn_without_glasses.config(state=tk.DISABLED if is_active else tk.NORMAL)
        self.btn_stop.config(state=tk.NORMAL if is_active else tk.DISABLED)
        self.btn_train.config(state=tk.DISABLED if is_active else tk.NORMAL)
        self.btn_test.config(state=tk.DISABLED if is_active else tk.NORMAL)
    
    def capture_image(self, frame, face_roi):
        if time.time() - self.last_capture < self.capture_interval:
            return False
            
        timestamp = int(time.time())
        if self.current_mode == 'with_glasses':
            img_name = f"with_glasses_{timestamp}.jpg"
            save_path = os.path.join(self.detector.with_glasses_dir, img_name)
        else:
            img_name = f"without_glasses_{timestamp}.jpg"
            save_path = os.path.join(self.detector.without_glasses_dir, img_name)
        
        cv2.imwrite(save_path, face_roi)
        self.last_capture = time.time()
        self.status_var.set(f"Saved: {img_name}")
        return True
    
    def update_video(self):
        if not hasattr(self, 'cap') or self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return
        
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.detector.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Get face ROI
                roi_gray = gray[y:y+h, x:x+w]
                
                # If capturing, save the face image
                if self.is_capturing and self.current_mode:
                    self.capture_image(frame, roi_gray)
                
                # Detect glasses
                glasses = self.detector.glasses_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
                
                # Draw status text
                status = "Glasses Detected" if len(glasses) > 0 else "No Glasses Detected"
                color = (0, 255, 0) if len(glasses) > 0 else (0, 0, 255)
                cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Convert to PhotoImage
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            
            # Update canvas
            self.canvas.config(width=frame.shape[1], height=frame.shape[0])
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img
        
        # Schedule next update
        self.root.after(10, self.update_video)
    
    def train_model(self):
        try:
            self.status_var.set("Training model...")
            self.root.update()
            
            # Call the detector's train method
            self.detector.train_model()
            
            # Count training samples
            with_glasses = len(os.listdir(self.detector.with_glasses_dir))
            without_glasses = len(os.listdir(self.detector.without_glasses_dir))
            
            self.status_var.set(f"Training complete! Samples: {with_glasses} with glasses, {without_glasses} without")
            messagebox.showinfo("Success", "Model trained successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
            self.status_var.set("Error during training")
    
    def test_detection(self):
        if not os.path.exists('glasses_model.yml'):
            messagebox.showerror("Error", "Model not found. Please train the model first.")
            return
        
        try:
            self.detector.recognizer.read('glasses_model.yml')
            self.status_var.set("Testing detection - Press 'q' to stop")
            
            cap = cv2.VideoCapture(0)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    roi_gray = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                    
                    # Predict
                    label, confidence = self.detector.recognizer.predict(roi_gray)
                    
                    if label == 0:  # With glasses
                        result = f"Glasses: Yes ({confidence:.2f})"
                        color = (0, 255, 0)  # Green
                    else:  # Without glasses
                        result = f"Glasses: No ({confidence:.2f})"
                        color = (0, 0, 255)  # Red
                    
                    cv2.putText(frame, result, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                cv2.imshow('Glasses Detection - Testing', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            self.status_var.set("Test mode ended")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to test detection: {str(e)}")
            self.status_var.set("Error during testing")
    
    def on_closing(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

class GlassesDetector:
    def __init__(self):
        # Initialize classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        
        # Create directories for training data
        self.base_dir = 'glasses_training_data'
        self.with_glasses_dir = os.path.join(self.base_dir, 'with_glasses')
        self.without_glasses_dir = os.path.join(self.base_dir, 'without_glasses')
        
        # Create directories if they don't exist
        os.makedirs(self.with_glasses_dir, exist_ok=True)
        os.makedirs(self.without_glasses_dir, exist_ok=True)
        
        # Initialize LBPH face recognizer
        if FACE_AVAILABLE:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        else:
            raise ImportError("cv2.face module is required. Please install opencv-contrib-python")
    
    def train_model(self):
        """Train the model using the collected data"""
        faces = []
        labels = []
        
        # Label 0 for with glasses, 1 for without glasses
        # Load with glasses images
        for img_name in os.listdir(self.with_glasses_dir):
            img_path = os.path.join(self.with_glasses_dir, img_name)
            img = Image.open(img_path).convert('L')
            img_np = np.array(img, 'uint8')
            faces.append(img_np)
            labels.append(0)  # 0 for with glasses
        
        # Load without glasses images
        for img_name in os.listdir(self.without_glasses_dir):
            img_path = os.path.join(self.without_glasses_dir, img_name)
            img = Image.open(img_path).convert('L')
            img_np = np.array(img, 'uint8')
            faces.append(img_np)
            labels.append(1)  # 1 for without glasses
        
        if len(faces) == 0:
            raise ValueError("No training data found! Please collect data first.")
        
        self.recognizer.train(faces, np.array(labels))
        self.recognizer.save('glasses_model.yml')

def main():
    root = tk.Tk()
    app = GlassesDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
