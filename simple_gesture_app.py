import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Global variables
bg = None
start_recording = False

# Region of Interest (ROI) coordinates
top, right, bottom, left = 10, 350, 225, 590

def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # Get the contours
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    else:
        # Find the largest contour
        segmented = max(contours, key=cv2.contourArea)
        return (thresholded, segmented)

def create_model():
    # Use Input layer as recommended
    inputs = layers.Input(shape=(89, 100, 1))
    
    # Build the model using functional API
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(3, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def getPredictedClass(model, image_path):
    # Predict
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Reshape for TensorFlow 2.x model
    input_data = gray_image.reshape(1, 89, 100, 1).astype('float32') / 255.0
    prediction = model.predict(input_data, verbose=0)
    return np.argmax(prediction[0]), np.max(prediction[0])

def resizeImage(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 89))
    cv2.imwrite(image_path, img)

def showStatistics(predictedClass, confidence):
    textImage = np.zeros((300,512,3), np.uint8)
    className = ""

    if predictedClass == 0:
        className = "Swing"
    elif predictedClass == 1:
        className = "Palm"
    elif predictedClass == 2:
        className = "Fist"

    cv2.putText(textImage,"Predicted Class : " + className, 
    (30, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)

    cv2.putText(textImage,"Confidence : " + str(confidence * 100) + '%', 
    (30, 100), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)
    cv2.imshow("Statistics", textImage)

def main():
    global start_recording
    
    # Initialize model
    model = create_model()
    print("Model created successfully.")
    
    # Create some dummy data for quick training
    dummy_x = np.random.random((100, 89, 100, 1))
    dummy_y = np.random.randint(0, 3, size=(100, 1))
    dummy_y = tf.keras.utils.to_categorical(dummy_y, 3)
    
    # Train for just 1 epoch to initialize weights
    print("Initializing model with random weights...")
    model.fit(dummy_x, dummy_y, epochs=1, batch_size=10, verbose=0)
    print("Model initialized with random weights.")
    
    # Initialize camera
    print("Starting camera...")
    camera = None
    
    # Try different camera backends
    for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
        try:
            camera = cv2.VideoCapture(0, backend)
            if camera.isOpened():
                print(f"Camera opened successfully with backend {backend}")
                break
        except Exception as e:
            print(f"Failed to open camera with backend {backend}: {e}")
    
    if camera is None or not camera.isOpened():
        print("Could not open camera. Exiting.")
        return
    
    # Initialize variables
    num_frames = 0
    aWeight = 0.5
    
    print("Starting gesture recognition. Press 'q' to quit, 's' to start/stop recording.")
    
    # Main loop
    while True:
        # Get the current frame
        (grabbed, frame) = camera.read()
        
        if not grabbed:
            print("Failed to grab frame. Exiting.")
            break
            
        # Flip the frame
        frame = cv2.flip(frame, 1)
        
        # Clone the frame
        clone = frame.copy()
        
        # Get the ROI
        roi = frame[top:bottom, right:left]
        
        # Convert the ROI to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Update the background model
        if num_frames < 30:
            run_avg(gray, aWeight)
            cv2.putText(clone, "Calibrating...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Segment the hand region
            hand = segment(gray)
            
            # Check if hand is segmented
            if hand is not None:
                # Unpack the thresholded image and segmented region
                (thresholded, segmented) = hand
                
                # Draw the segmented region
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                
                # Process the hand if recording is on
                if start_recording:
                    cv2.imwrite('Temp.png', thresholded)
                    resizeImage('Temp.png')
                    predictedClass, confidence = getPredictedClass(model, 'Temp.png')
                    showStatistics(predictedClass, confidence)
                
                # Show the thresholded image
                cv2.imshow("Thresholded", thresholded)
        
        # Draw the ROI rectangle
        cv2.rectangle(clone, (right, top), (left, bottom), (0, 255, 0), 2)
        
        # Increment the number of frames
        num_frames += 1
        
        # Display the frame
        cv2.imshow("Video Feed", clone)
        
        # Check for key presses
        keypress = cv2.waitKey(1) & 0xFF
        
        if keypress == ord("q"):
            break
        elif keypress == ord("s"):
            start_recording = not start_recording
            print("Recording:", "ON" if start_recording else "OFF")
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()