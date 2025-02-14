from PIL import Image, ImageDraw
import numpy as np
import time


def solve(color_positions, board):
    if len(color_positions) == 0:  # If no colors left we are done
        return True
    else:
        for pos in color_positions[0]:  # For all positions of first color in list
            if board[pos[0], pos[1]] == 0:  # If unused
                new_board = board.copy()
                place_queen(pos, new_board)  # Place Queen
                if solve(color_positions[1:], new_board):  # and solve for remaining colors
                    board[:] = new_board  # passing back solution if solved
                    return True
    return False


def place_queen(position, board):
    size = board.shape[0]
    board[position[0], position[1]] = 1  # Mark given position
    for n in range(size):  # Lock out same row and column
        if board[position[0], n] == 0:
            board[position[0], n] = -1
        if board[n, position[1]] == 0:
            board[n, position[1]] = -1
    for row_offset in [-1, 1]:  # Corners, taking care of edges
        current_row = 0 if position[0] + row_offset < 0 else size - 1 if position[0] + row_offset > size - 1 else position[0] + row_offset
        for col_offset in [-1, 1]:
            current_col = 0 if position[1] + col_offset < 0 else size - 1 if position[1] + col_offset > size - 1 else position[1] + col_offset
            if board[current_row, current_col] == 0:
                board[current_row, current_col] = -1
    return


def load_image(image_path):
    """Load the puzzle image (PNG file)."""
    return Image.open(image_path)


def grid_count(img):
    np_img = np.array(img.convert("L"))  # Convert to grayscale array
    np_img[np_img > 100] = 255  # Binary threshold
    np_img[np_img <= 100] = 0

    img_width, img_height = np_img.shape
    vertical_count = 0  # Vertical count (array, not image!)
    horizontal_count = 0  # Horizontal count
    color_run_threshold = 20  # Threshold of color runs
    run_count = 0
    while vertical_count == 0:  # Repeat until midline not black
        random_y = np.random.randint(10, img_height - 20)  # Random midpoint
        for x in range(img_width):  # Move vertical
            if np_img[x, random_y]:  # Color found
                run_count += 1  # Increase run count
            else:  # Black line found
                if run_count >= color_run_threshold:  # Last color run over threshold?
                    vertical_count += 1  # Then we have a square
                run_count = 0  # Reset count
    run_count = 0
    while horizontal_count == 0:  # Repeat until midline not black
        random_x = np.random.randint(10, img_width - 20)  # Random midpoint
        for y in range(img_height):  # Move horizontal
            if np_img[random_x, y]:  # Color found
                run_count += 1  # Increase run count
            else:  # Black line found
                if run_count >= color_run_threshold:  # Last color run over threshold?
                    horizontal_count += 1  # Then we have a square
                run_count = 0  # Reset count

    return max(vertical_count, horizontal_count)


def code_board(img):
    img_width, img_height = img.size

    grid_size = grid_count(img)
    print(f"Detected grid size: {grid_size}x{grid_size}")

    # Identify solid colors and discard color compression artifacts:
    colors = img.getcolors(maxcolors=10000)  # Get pixel count per color (overkill limit)
    colors.sort(key=lambda x: 1e10 if sum(x[1]) == 0 else x[0], reverse=True)  # Sort in descending order, keep black first
    colors = colors[:grid_size + 1]  # Keep "solid" colors (most frequent)
    colors = colors[1:]  # Drop black
    color_dict = dict()
    for i in range(len(colors)):  # Color dictionary
        color_dict[colors[i][1]] = i  # Code each RGB with an index
    color_dict[(0, 0, 0)] = grid_size  # Recover black with code = color count

    square_size = img_width // grid_size  # Approx. square size

    color_board = np.zeros((grid_size, grid_size), dtype=np.int8)  # Empty colors board

    for row in range(grid_size):  # For each row
        sample_y = square_size // 2 + row * square_size
        for col in range(grid_size):  # and each column
            sample_x = square_size // 2 + col * square_size  # get approx midpoint
            increment = 1
            color_code = -1
            while color_code == -1:  # While we don't get a recognized color (compression artifacts get in the middle)
                sample_x += increment  # shift sample point one pixel
                color_code = color_dict.get(img.getpixel((sample_x, sample_y)), -1)  # and sample again
                if color_code == grid_size:  # if reached the edge of the square revert shift direction
                    sample_x -= increment
                    increment = -increment
                    grid_size = -1
            color_board[row, col] = color_code  # Assign color code for current row and column to board representation

    return color_board


def color_lists(board):
    size = board.shape[0]
    color_positions = [[] for n in range(size)]  # List of empty lists, one per color
    for row in range(size):  # For all rows
        for col in range(size):  # For all columns
            color_positions[board[row, col]].append((row, col))  # add square position to color list

    color_positions.sort(key=lambda x: len(x))  # Sort list by number of squares of each color
    return color_positions


def apply_solution(board, img, queen_img):
    grid_size = board.shape[0]  # row-col count
    square_size = img.size[0] // grid_size  # approx square size

    # Resize the queen image to fit within a grid square
    queen_img = queen_img.resize((square_size, square_size), Image.LANCZOS)

    for row in range(grid_size):
        for col in range(grid_size):
            if board[row, col] == 1:
                # Calculate the position to paste the queen image
                position = (col * square_size, row * square_size)
                img.paste(queen_img, position, queen_img)  # Use the queen image as a mask to handle transparency

    return


if __name__ == '__main__':
    image_path = input("Enter the path to your puzzle image (e.g., 'puzzle.png'): ")

    img = load_image(image_path)  # Load the puzzle image
    queen_img = load_image("queen.png")  # Load the queen image

    img_width, img_height = img.size
    if abs(1 - img_width / img_height) > 0.02:
        print(1 - img_width / img_height, img.size)
        img.show()
        raise Exception("Wrong shape detected - verify your layout.")

    color_board = code_board(img)  # Color coded board
    color_positions = color_lists(color_board)  # Color position lists
    board = np.zeros(color_board.shape, dtype=np.int8)  # initial empty board

    if solve(color_positions, board):  # If solution can be found
        apply_solution(board, img, queen_img)  # Apply solution to image with queen images
        img.show()  # Show completed board
        print("Solved")
    else:
        print("Problem has no solution")
