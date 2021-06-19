import cv2
import numpy as np

img1In = cv2.imread("BinarizeSobelXY.png", cv2.IMREAD_GRAYSCALE)
img2In = cv2.imread("assinaturaFernando2Binarized.png", cv2.IMREAD_GRAYSCALE)

img1 = cv2.resize(img1In, (400, 400))
img2 = cv2.resize(img2In, (400, 400))

#orb detector
orb = cv2.ORB_create()
# key points and descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# descriptor é um array de numeros da imagem, e vão descrever a feature.
# Para cada feature encontrada na imagem, a gente tem um descriptor

# Como comparar as features? Não podemos comparar pixel a pixel pois pode ser diferente
# por causa de luminosidade, rotação etc

# Descriptors definem a feature independente desses fatores mostrados

# Vou comparar os descriptos da imagem 1 e da dois e terá um match usando brute force

# Brute force matching images
# Ele vai comparar cada descriptor da imagem 1 com cada descriptor da imagem 2
# Os mais parecidos vão ser considerados como um match de ok

# cross check vai definir sempre conseguir o melhor match, durante a comparação de features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(des1, des2)
# Ordenando numeros
matches = sorted(matches, key = lambda x:x.distance)

# print(len(matches))

# cada m é um objeto que contem informação 
# Quanto menor o m distance, melhor. Foi melhor detectado
goodValues: list = []
for m in matches:
  if(m.distance < 40):
    goodValues.append(m)  

if(len(goodValues) > 40):
  print('Imagens similares', len(goodValues))
else:
  print('Imagens diferentes', len(goodValues))  

# # Draw matchs
matching_result = cv2.drawMatches(img1, kp1, img2, kp2, goodValues, None, flags=2)

cv2.imshow("Image1", img1)
cv2.imshow("Image2", img2)
cv2.imshow("My Result", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

