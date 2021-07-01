import cv2
import numpy as np

class flag:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def Austria(self):
        image_size = (self.length,self.width,3)
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


    def Janpan(self):
        # flag-of-Japan
        image_size =  (self.length,self.width,3)
        I = (image_size[0]/2,image_size[1]/2)
        flag_of_Japan = np.ones((image_size[0],image_size[1],image_size[2]), dtype=np.uint8)
        for i in range(image_size[0]):
            for j in range(image_size[1]):
                if (I[0]-i)**2+(I[1]-j)**2 <=(image_size[0]/4)**2:
                    flag_of_Japan[i][j] = np.asarray([0, 0, 255])
                else:
                    flag_of_Japan[i][j] = np.asarray([255, 255, 255])

        cv2.imwrite(r"img\flag_of_Japan.png", flag_of_Japan)
        cv2.imshow('Image', flag_of_Japan)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Turkey(self):
        # flag_of_Turkey
        image_size =  (self.length,self.width,3)    #(120,160)  (240, 320)
        flag_of_Tuarkey = np.ones((image_size[0],image_size[1],image_size[2]), dtype=np.uint8)
        for i in range(image_size[0]):
            for j in range(image_size[1]):
                if (self.length/2 - i) ** 2 + (self.width/4 - j) ** 2 <= (self.width/8) ** 2 and ((self.length/2 - i) ** 2 + ((self.width/4+self.width/20) - j) ** 2 >= (self.width/8-self.width/20+5) ** 2):
                    flag_of_Tuarkey[i][j] = np.asarray([255, 255, 255])
                else:
                    flag_of_Tuarkey[i][j] = np.asarray([0, 0, 255])

        cv2.imwrite(r"img\flag_of_Turkey.png", flag_of_Tuarkey)
        cv2.imshow('Image', flag_of_Tuarkey)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Qatar(self):
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


if __name__ == '__main__':
    flags = flag(120*2,240*2)
    flags.Austria()
    flags.Janpan()
    flags.Turkey()
    flags.Qatar()