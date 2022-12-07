from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
img1 = cv2.imread("C:\Img.jpeg")
# img1 = cv2.imread("C:\Img2.jpeg")
# img1 = cv2.imread("C:\Img3.jpeg")
# img1 = cv2.imread("C:\Img4.jpeg")
plt.imshow(img1[:,:,::-1])
plt.show()
result = DeepFace.analyze(img1, actions = ['age','gender'])
print(result)

 