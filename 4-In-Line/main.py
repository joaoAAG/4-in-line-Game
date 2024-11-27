import cv2
import mediapipe as mp
import numpy as np
import math
import time



# Set up MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Constants for defining game board and cell size
COLS = 8
ROWS = 6
CELL_WIDTH = 60
CELL_HEIGHT = 60
GRID_WIDTH = COLS * CELL_WIDTH
GRID_HEIGHT = ROWS * CELL_HEIGHT
X_OFFSET = (640 - GRID_WIDTH) // 2
Y_OFFSET = (480 - GRID_HEIGHT) // 2

# Load gear icon image
gear_icon_path = "gear_icon.png"  # Path to your gear icon image
gear_icon = cv2.imread(gear_icon_path, cv2.IMREAD_UNCHANGED)

# Initialize game board
board = np.zeros((ROWS, COLS))
player_turn = 1
game_over = False
selected_column = 0  # Initialize selected column

def map_hand_to_column(hand_x):
    # Map hand x-coordinate to column index
    return min(max(int((hand_x - X_OFFSET) / CELL_WIDTH), 0), COLS - 1)

def drop_token(column):
    for row in range(ROWS - 1, -1, -1):
        if board[row][column] == 0:
            board[row][column] = player_turn
            return row, column
    return None
def draw_grid(frame):
    # Draw grid lines
    for i in range(COLS + 1):
        cv2.line(frame, (X_OFFSET + i * CELL_WIDTH, Y_OFFSET),
                 (X_OFFSET + i * CELL_WIDTH, Y_OFFSET + GRID_HEIGHT),
                 (255, 255, 255), 2)
    for j in range(ROWS + 1):
        cv2.line(frame, (X_OFFSET, Y_OFFSET + j * CELL_HEIGHT),
                 (X_OFFSET + GRID_WIDTH, Y_OFFSET + j * CELL_HEIGHT),
                 (255, 255, 255), 2)

    # Highlight selected column
    cv2.rectangle(frame, (X_OFFSET + selected_column * CELL_WIDTH, Y_OFFSET),
                  (X_OFFSET + (selected_column + 1) * CELL_WIDTH, Y_OFFSET + GRID_HEIGHT),
                  (0, 255, 0), 2)

    # Resize and draw gear icon
    gear_icon_resized = cv2.resize(gear_icon, (40, 40))  # Resize gear icon
    gear_icon_resized_bgr = gear_icon_resized[:, :, :3]  # Remove alpha channel
    frame[Y_OFFSET - 50:Y_OFFSET - 10, X_OFFSET + GRID_WIDTH - 50:X_OFFSET + GRID_WIDTH - 10] = gear_icon_resized_bgr

def check_win(board, player):
    # Check rows
    for row in range(ROWS):
        for col in range(COLS - 3):
            if board[row][col] == player and board[row][col + 1] == player and board[row][col + 2] == player and \
                    board[row][col + 3] == player:
                return True

    # Check columns
    for col in range(COLS):
        for row in range(ROWS - 3):
            if board[row][col] == player and board[row + 1][col] == player and board[row + 2][col] == player and \
                    board[row + 3][col] == player:
                return True

    # Check diagonals (top-left to bottom-right)
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            if board[row][col] == player and board[row + 1][col + 1] == player and board[row + 2][col + 2] == player and \
                    board[row + 3][col + 3] == player:
                return True

    # Check diagonals (top-right to bottom-left)
    for row in range(ROWS - 3):
        for col in range(3, COLS):
            if board[row][col] == player and board[row + 1][col - 1] == player and board[row + 2][col - 2] == player and \
                    board[row + 3][col - 3] == player:
                return True

    return False

# Main loop
cap = cv2.VideoCapture(0)

# Delay between plays (in seconds)
play_delay = 2

# Time of the last play
last_play_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for selfie-view display
    frame = cv2.flip(frame, 1)

    # Process frame to detect hands
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and not game_over:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand landmarks
            hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]
            hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]
            thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]
            thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0]

            # Check if thumb and index finger are pinching
            if math.sqrt((hand_x - thumb_x) ** 2 + (hand_y - thumb_y) ** 2) < 30:
                # Map hand x-coordinate to column index
                selected_column = map_hand_to_column(hand_x)

                # Check if it's time for the next play
                current_time = time.time()
                if current_time - last_play_time >= play_delay:
                    # Drop token into selected column
                    play_result = drop_token(selected_column)
                    if play_result:
                        # Check for win
                        if check_win(board, player_turn):
                            print("Player", player_turn, "wins!")
                            game_over = True
                        else:
                            # Switch player turn
                            player_turn = 2 if player_turn == 1 else 1
                            last_play_time = current_time
                            print("Player", player_turn, "plays.")

                            # Uncomment this line if you want to see the updated board after each play
                            # print(board)

    # Draw grid lines and highlight selected column on frame
    draw_grid(frame)

    # Draw tokens on frame
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] == 1:
                cv2.circle(frame,
                           (X_OFFSET + c * CELL_WIDTH + CELL_WIDTH // 2, Y_OFFSET + r * CELL_HEIGHT + CELL_HEIGHT // 2),
                           min(CELL_WIDTH, CELL_HEIGHT) // 3, (0, 0, 255), -1)
            elif board[r][c] == 2:
                cv2.circle(frame,
                           (X_OFFSET + c * CELL_WIDTH + CELL_WIDTH // 2, Y_OFFSET + r * CELL_HEIGHT + CELL_HEIGHT // 2),
                           min(CELL_WIDTH, CELL_HEIGHT) // 3, (0, 255, 255), -1)

    # Display frame
    cv2.imshow('Connect Four', frame)

    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
cap.release()
