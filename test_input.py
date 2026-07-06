import pydirectinput
import time

print("Opening Notepad — click into it now, you have 5 seconds...")
time.sleep(5)
pydirectinput.press('a')
pydirectinput.press('a')
pydirectinput.press('a')
print("Done — did 'aaa' get typed into Notepad?")