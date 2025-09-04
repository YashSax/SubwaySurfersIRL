import subprocess
from pynput import keyboard

# --- Start persistent adb shell ---
adb = subprocess.Popen(
    ["adb", "shell"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
)

def send_cmd(cmd: str):
    """Send a command to the persistent adb shell."""
    adb.stdin.write(cmd + "\n")
    adb.stdin.flush()

# --- Game controls ---
def jump():
    send_cmd("input swipe 500 1200 500 600 1")   # fast upward swipe

def roll():
    send_cmd("input swipe 500 600 500 1200 1")   # fast downward swipe

def left():
    send_cmd("input swipe 800 1000 300 1000 1")  # fast left swipe

def right():
    send_cmd("input swipe 300 1000 800 1000 1")  # fast right swipe

# --- Keyboard listener ---
def on_press(key):
    try:
        if key == keyboard.Key.up:
            print("Jump")
            jump()
        elif key == keyboard.Key.down:
            print("Roll")
            roll()
        elif key == keyboard.Key.left:
            print("Left")
            left()
        elif key == keyboard.Key.right:
            print("Right")
            right()
    except Exception as e:
        print("Error:", e)

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener when ESC is pressed
        return False

print("Controller ready! Use arrow keys to play Subway Surfers. Press ESC to quit.")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
