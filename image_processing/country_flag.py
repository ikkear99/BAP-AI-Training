import cv2
import numpy as np

image_size = (120,180,3)
# flag-of-Austria
flag_of_Austria = np.ones((image_size[0],image_size[1],image_size[2]), dtype=np.uint8)
for i in range(image_size[0]):
    for j in range(image_size[1]):
        if i < image_size[0]/3:
            flag_of_Austria[i][j] = np.asarray([46, 0, 255])
        elif (i < image_size[0]*2/3) and (i >=image_size[0]/3):
            flag_of_Austria[i][j] = np.asarray([255, 255, 255])
        else :
            flag_of_Austria[i][j] = np.asarray([46, 0, 255])

cv2.imwrite(r"img\flag_of_Austria.png", flag_of_Austria)
cv2.imshow('Image', flag_of_Austria)
cv2.waitKey(0)
cv2.destroyAllWindows()



# flag-of-Japan
I = (image_size[0]/2,image_size[1]/2)
flag_of_Japan = np.ones((image_size[0],image_size[1],image_size[2]), dtype=np.uint8)
for i in range(image_size[0]):
    for j in range(image_size[1]):
        if (I[0]-i)**2+(I[1]-j)**2 <=30**2:
            flag_of_Japan[i][j] = np.asarray([0, 0, 255])
        else:
            flag_of_Japan[i][j] = np.asarray([255, 255, 255])

cv2.imwrite(r"img\flag_of_Japan.png", flag_of_Japan)
cv2.imshow('Image', flag_of_Japan)
cv2.waitKey(0)
cv2.destroyAllWindows()



# flag_of_Turkey
I = (60,50,3)
I1 = (60, 55,3)
flag_of_Japan = np.ones((image_size[0],image_size[1],image_size[2]), dtype=np.uint8)
for i in range(image_size[0]):
    for j in range(image_size[1]):
        if (60-i)**2+(50-j)**2 <=30**2 and ((60-i)**2+(55-j)**2 >= 26**2):
            flag_of_Japan[i][j] = np.asarray([255, 255, 255])
        else:
            flag_of_Japan[i][j] = np.asarray([0, 0, 255])

cv2.imwrite(r"img\flag_of_Turkey.png", flag_of_Japan)
cv2.imshow('Image', flag_of_Japan)
cv2.waitKey(0)
cv2.destroyAllWindows()


# flag_of_Quatar
image_size = (117,180,3)
flag_of_Japan = np.ones((image_size[0],image_size[1],image_size[2]), dtype=np.uint8)
for i in range(117):
    for j in range(180):
        if j <60:
            flag_of_Japan[i][j] = np.asarray([255, 255, 255])
        elif j >=60:
            flag_of_Japan[i][j] = np.asarray([18, 18, 138])

j = 0
dem = 0

while j < 117:
    for k in range(60, 60 + dem):
        flag_of_Japan[j][k] = np.asarray([255, 255, 255])
    dem += 3
    if dem == 21:
        dem = 18
        while dem != 0:
            for k in range(60, 60 + dem):
                flag_of_Japan[j][k] = np.asarray([255, 255, 255])
            j+=1
            dem -= 3
    j+=1


cv2.imwrite(r"img\flag_of_Quatar.png", flag_of_Japan)
cv2.imshow('Image', flag_of_Japan)
cv2.waitKey(0)
cv2.destroyAllWindows()




