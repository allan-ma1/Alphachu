from pynput.mouse import Button, Controller

mouse = Controller()

def mouse_click():
    mouse.position = (500, 400)
    mouse.press(Button.left)
    mouse.release(Button.right)