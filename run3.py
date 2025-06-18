from pynput import mouse

def on_scroll(x, y, dx, dy):
    if dy > 0:
        print("Scrolled Up")
    elif dy < 0:
        print("Scrolled Down")

# Start the listener in blocking mode (this will keep the script running)
with mouse.Listener(on_scroll=on_scroll) as listener:
    print("Listening for scroll gestures... (Press Ctrl+C to stop)")
    listener.join()  # Keeps the script running
def on_scroll(x, y, dx, dy):
    print(f"Scroll event detected at ({x}, {y}), dx={dx}, dy={dy}")
    if dy > 0:
        print("Scrolled Up")
    elif dy < 0:
        print("Scrolled Down")
