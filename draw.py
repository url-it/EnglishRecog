import pygame
import sys
import numpy as np
from tensorflow.keras.models import load_model

# Initialize Pygame
pygame.init()

# Set up display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Drawing and Recognition Program")
screen.fill((255, 255, 255))

# Set up colors
black = (0, 0, 0)
white = (255, 255, 255)

# Set up drawing variables
drawing = False
radius = 5
color = white
last_pos = (0, 0)

# Load the trained model
model = load_model('EnlgishRecModel.h5')

# Font for displaying the predicted letter
font = pygame.font.Font(None, 36)

# Captured image and predicted letter display
captured_image = None
predicted_label_text = None

def capture_image():
    global captured_image, predicted_label_text

    # Capture the drawn image
    pygame.image.save(screen.subsurface((0, 0, width, height)), 'drawn_image.png')

    # Preprocess the image
    drawn_image = pygame.image.load('drawn_image.png').convert()
    drawn_image = pygame.transform.scale(drawn_image, (28, 28))  # Assuming input shape of your model is (28, 28, 1)
    drawn_image_array = pygame.surfarray.array3d(drawn_image)
    drawn_image_array = np.mean(drawn_image_array, axis=2)  # Convert to grayscale
    drawn_image_array = np.expand_dims(drawn_image_array, axis=0)
    drawn_image_array = np.expand_dims(drawn_image_array, axis=-1)
    drawn_image_array = drawn_image_array / 255.0  # Normalize to [0, 1]

    # Make prediction using the model
    prediction = model.predict(drawn_image_array)
    predicted_label = np.argmax(prediction)
    print("Predicted Label:", predicted_label)

    # Display the captured image and predicted letter on the side
    captured_image = pygame.image.load('drawn_image.png')
    predicted_label_text = font.render("Predicted: {}".format(chr(predicted_label + 65)), True, black)

# Erase button
erase_button_rect = pygame.Rect(width + 10, height // 4, 120, 40)
erase_button_color = (255, 0, 0)

# Capture button
capture_button_rect = pygame.Rect(width + 10, 2 * height // 4, 120, 40)
capture_button_color = (0, 255, 0)

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                drawing = True
                last_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False

        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                pygame.draw.line(screen, color, last_pos, event.pos, radius)
                last_pos = event.pos

        # Check button clicks
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if erase_button_rect.collidepoint(event.pos):
                # Erase button clicked
                screen.fill(white)
                captured_image = None
                predicted_label_text = None
            elif capture_button_rect.collidepoint(event.pos):
                # Capture button clicked
                capture_image()

        # Check key presses
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                # 'C' key pressed
                capture_image()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                # 'E' key pressed
                screen.fill(white)
                captured_image = None
                predicted_label_text = None

    # Draw buttons
    pygame.draw.rect(screen, erase_button_color, erase_button_rect)
    erase_text = font.render("Erase", True, white)
    screen.blit(erase_text, (width + 30, height // 4 + 10))

    pygame.draw.rect(screen, capture_button_color, capture_button_rect)
    capture_text = font.render("Capture", True, white)
    screen.blit(capture_text, (width + 20, 2 * height // 4 + 10))

    # Draw the captured image and predicted letter
    if captured_image is not None:
        screen.blit(captured_image, (width + 10, 3 * height // 4))
        if predicted_label_text is not None:
            screen.blit(predicted_label_text, (width + 10, 3 * height // 4 + 160))

    # Update the display
    pygame.display.flip()

    # Set the frame rate
    pygame.time.Clock().tick(60)