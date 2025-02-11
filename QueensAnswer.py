import cv2
import numpy as np

def detect_board_and_colors(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get a binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty board mask
    board = np.zeros_like(image)

    # Draw the board as a filled polygon
    cv2.drawContours(board, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Convert image to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Detect unique colors dynamically
    unique_colors = {}
    for y in range(0, hsv.shape[0], 10):
        for x in range(0, hsv.shape[1], 10):
            pixel = tuple(hsv[y, x])
            if pixel not in unique_colors:
                unique_colors[pixel] = 1
            else:
                unique_colors[pixel] += 1

    # Filter out less frequent colors (noise)
    threshold = 50
    detected_colors = {k: v for k, v in unique_colors.items() if v > threshold}

    # Create a blank board with detected colors
    reconstructed_board = np.zeros_like(image)

    cell_size = image.shape[0] // 8  # Assuming an 8x8 grid
    for y in range(8):
        for x in range(8):
            pixel_pos = (y * cell_size + cell_size // 2, x * cell_size + cell_size // 2)
            color_pixel = hsv[pixel_pos[0], pixel_pos[1]]
            closest_color = min(detected_colors.keys(), key=lambda c: np.linalg.norm(np.array(c) - np.array(color_pixel)))

            # Convert HSV to BGR before drawing
            closest_color_bgr = cv2.cvtColor(np.uint8([[closest_color]]), cv2.COLOR_HSV2BGR)[0][0].tolist()

            cv2.rectangle(reconstructed_board, (x * cell_size, y * cell_size),
                          ((x + 1) * cell_size, (y + 1) * cell_size),
                          closest_color_bgr, thickness=-1)

    # Draw empty squares for potential queen placements
    for y in range(8):
        for x in range(8):
            cv2.rectangle(reconstructed_board, (x * cell_size, y * cell_size),
                          ((x + 1) * cell_size, (y + 1) * cell_size),
                          (0, 0, 0), thickness=2)  # Black border for empty squares

    # Display reconstructed board
    cv2.imshow("Reconstructed Board", reconstructed_board)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_board_and_colors("queens_screenshot.png")
