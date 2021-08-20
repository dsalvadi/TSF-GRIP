import numpy as np
import pandas as pd
import cv2

# read image
img = cv2.imread("image_1.jpg")
h, w = img.shape[0], img.shape[1]

# resize image to 1000 px
scale = 1000 / max(h,w)
dim = (int(w*scale), int(h*scale))
# print(dim, scale)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


# read csv file containing color daya
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names = index, header = None)

# initialize global variables
clicked = False
r = g = b = xpos = ypos = 0

# define function for recognizing color
def recognize_color(R, G, B):
    minimum = 10000
    for i in range (len(csv)):
        d = abs(R - int(csv.loc[i,"R"])) + abs(G - int(csv.loc[i,"G"])) + abs(B - int(csv.loc[i,"B"]))
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname

# define function to detect a double mouse click
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, xpos, ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b, g, r = img[y,x]
        b = int(b)
        g = int(g)
        r = int(r)

# Create Application Window
cv2.namedWindow("Color Identifier")
cv2.setMouseCallback("Color Identifier", mouse_click)

while True:
    cv2.imshow("Color Identifier", img)
    if clicked:
        # cv2.rectangle(image, startpoint, endpoint, color, thickness)-1 fills entire rectangle 
        cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)
        
        # creating text string to display( Color name and RGB values )
        text = recognize_color(r, g, b) + ' R='+ str(r) + ' G='+ str(g) + ' B=' + str(b)
        
        # cv2.putText(img, text, start, font(0-7), fontScale,color,thickness,lineType )
        cv2.putText(img, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # for very light colours we will display text in black colour
        if(r+g+b>=600):
            cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            
        clicked = False
        
    # break loop when 'esc' is pressed
    if cv2.waitKey(20) & 0xFF ==27:
        break

cv2.destroyAllWindows()