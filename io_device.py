import pyautogui as p

def drag_cursor(x=0,y=0):
    p.moveTo(x,y)


def click_mouse_down():
    p.mouseDown()

    
def click_mouse_up():
    p.mouseUp()