import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

from numpy.lib.function_base import angle


image = cv2.imread("assvermelha.png")
image2= cv2.imread("teste1.jpg")
image3= cv2.imread("asspreta.png")

def proc(image):

        #Imagem 1
    #Altura
    H = image.shape[0]
    #Largura
    W = image.shape[1]


    #Criação de matriz com mesmo tamanho da imagem tendo todos os elementos zerados
    grayImg = np.zeros((H,W), np.uint8)
    #Cálculo do valor de cinza a partir da formula Cinza = Azul*0.07 + Verde*0.72 + Vermelho*0.21
    for i in range(H):
        for j in range(W):
            grayImg[i,j] = np.clip(0.07 * image[i,j,0]  + 0.72 * image[i,j,1] + 0.21 * image[i,j,2], 0, 255)





    ############################ FILTRO DE MEDIA ############################
    def MediaFilter(grayImg):

            #Definindo a dimensão que os lados do filtro terão
            filter_dim = 3
            den = filter_dim*filter_dim

            #Cria uma matriz de 0s com as dimensões definidas anteriormente
            filter_Media = np.zeros((filter_dim, filter_dim), np.float32)


            #Cálculo do filtro
            for x in range(filter_dim):
                    for y in range(filter_dim):
                            filter_Media[x,y] = 1/den

            
            imgFiltrada = np.zeros_like(grayImg, dtype=np.float32)

            imgFiltrada = convolution(grayImg, filter_Media, filter_dim)

            return (imgFiltrada.astype(np.uint8))

    ############################ FILTRO GAUSSIANO ############################

    def GaussianFilter(grayImg, sigma):
        
        #Definindo a dimensão que os lados do filtro terão
        filter_dim = 3
        imgFiltrada = cv2.GaussianBlur(grayImg,(3,3),sigma)

            #Cria uma matriz de 0s com as dimensões definidas anteriormente
            #     filter_Gauss = np.zeros((filter_dim, filter_dim), np.float32)
            #     d = filter_dim//2
            
            #     #Cálculo do filtro
            #     for x in range(-d, d+1):
            #         for y in range(-d, d+1):
            #             filter_Gauss[x+d, y+d] = (1/(2*np.pi*(sigma**2)))*(np.exp(-(x**2 + y**2)/(2* sigma**2)))

            #     print (filter_Gauss)
            #     imgFiltrada = np.zeros_like(grayImg, dtype=np.float32)
            
            #     imgFiltrada = convolution(grayImg, filter_Gauss, filter_dim)
        
        return (imgFiltrada.astype(np.uint8))


    ############################ FILTRO DE FOURIER ############################
    def FourierFilter(grayImg, r):

            # Fourier transforma a imagem, fft é uma matriz tridimensional, fft [:,:, 0] é a parte real, fft [:,:, 1] é a parte imaginária
        fft = cv2.dft(np.float32(grayImg), flags=cv2.DFT_COMPLEX_OUTPUT)
            # Centraliza fft, o dshift gerado ainda é uma matriz tridimensional
        dshift = np.fft.fftshift(fft)

            # Obtenção do pixel central
        rows, cols = grayImg.shape[:2]
        mid_row, mid_col = int(rows / 2), int(cols / 2)

            # Construção da máscara
        mask = np.zeros((rows, cols, 2), np.float32)
        mask[mid_row - r:mid_row + r, mid_col - r:mid_col + r] = 1

            
        fft_filtering = dshift * mask
                # Transformada inversa
        ishift = np.fft.ifftshift(fft_filtering)
        image_filtering = cv2.idft(ishift)
        image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
                # Normaliza os resultados da transformação inversa 
        cv2.normalize(image_filtering, image_filtering, 0, 1, cv2.NORM_MINMAX)
        return image_filtering
    
    

    ############################ FUNÇÃO DE CONVOLUÇÃO ############################

    def convolution(original, kernel, kernel_dim):
        
        image_pad = np.pad(original, pad_width=((kernel_dim // 2, kernel_dim // 2),(kernel_dim // 2, kernel_dim // 2)), mode='constant', constant_values=0).astype(np.float32)
        
        
        kd = kernel_dim // 2
        
        image_conv = np.zeros(image_pad.shape)
        
        for i in range(kd, image_pad.shape[0]-kd):
            for j in range(kd, image_pad.shape[1]-kd):
                x = image_pad[i-kd:i-kd+kernel_dim, j-kd:j-kd+kernel_dim]
                x = x.flatten()*kernel.flatten()

                image_conv[i][j] = x.sum()
        return image_conv

    
    #Gerando novas imagens após a cinza original passar pelos Filtros Passa Baixa
    imGaussian = GaussianFilter(grayImg, 1)   
    imMedia = MediaFilter(grayImg)
    imFourier = FourierFilter(grayImg, 100)

    ##################### FILTRO DE SOBEL ######################

    def SobelFilter(image):
        # filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        # filter_dim = 3

        filter = np.array([[2,2,4,2,2], [1,1,2,1,1],[0,0,0,0,0],[-1,-1,-2, -1,-1],[-2,-2,-4,-2,-2]])
        filter_dim = 5

        kd = filter_dim//2

        imSV = convolution(image,filter,filter_dim)
        imSH = convolution(image,filter.T,filter_dim)
        
        imgFiltrada = np.zeros_like(image, dtype=np.float32)
        d = np.zeros_like(image, dtype=np.float32)

        for i in range(H):
            for j in range(W):
                imgFiltrada[i,j]= image[i,j]
                d[i,j] = np.hypot(imSV[i,j],imSH[i,j])
                if d[i,j] > 90.0:
                    imgFiltrada[i,j] = image[i,j]*0.7
        # print("Im", image.shape)        
        # print("S",imgFiltrada.shape)
        return imgFiltrada.astype(np.uint8), filter_dim

    ##################### FILTRO LAPLACIANO ######################

    def LaplaceFilter(image):

        # filter = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
        # filter = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
        filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filter_dim = 3
        
        
        img = convolution(image,filter,filter_dim)

        imgFiltrada = np.zeros_like(image, dtype=np.float32)

        for i in range(H):
            for j in range(W):
                if image[i,j] > 245:
                        imgFiltrada[i,j] = 255
                else:
                    imgFiltrada[i,j]= 255 - img[i,j]
                # imgFiltrada[i,j] = img[i,j]
                # imgFiltrada[i,j] = img[i,j]
            
                # elif img[i,j]<=255:
                #     imgFiltrada[i,j] = 255
            
                    
                # if ((img[i,j] < 0) or (img[i,j] > 10)):
                #     imgFiltrada[i,j] = 255
                # else:
                #     imgFiltrada[i,j] = 0
        print("L", imgFiltrada.shape)
        return imgFiltrada.astype(np.uint8), filter_dim
        

    #Gerando novas imagens aplicando Filtros Passa Alta na imagem suavizada anteriormente
    imSobel, dimS = SobelFilter(imMedia)
    imLaplace, dimL = LaplaceFilter(imMedia)

    ############################## BINARIZAÇÃO COM OTSU ################################
    def binaryImage(image,kd):

            

            #Cálculo do histograma e das bordas das colunas
            histog_val, bordas_col = np.histogram(image, 256)

            # Cálculo dos centros de cada coluna
            centros_col = (bordas_col[:-1] + bordas_col[1:]) / 2.

            #Cálculo do peso em cada coluna de forma cumulativa da Esquerda para a Direita
            peso_cresc = np.cumsum(histog_val)

            #Cálculo do peso em cada coluna de forma cumulativa da Direita para a Esquerda
            peso_decresc = np.cumsum(histog_val[::-1])[::-1]

            #Cálculo das médias
            media_cresc = np.cumsum(histog_val * centros_col) / peso_cresc
            media_decresc = (np.cumsum((histog_val * centros_col)[::-1]) / peso_decresc[::-1])[::-1]

            #Cálculo das variâncias interclasses
            variancia_interclasses = peso_cresc[:-1] * peso_decresc[1:] * (media_cresc[:-1] - media_decresc[1:]) ** 2

            #Cálculo do caso que tem a variância máxima
            max_variancia = np.argmax(variancia_interclasses)

            #Definição do melhor linear
            linear = centros_col[:-1][max_variancia]
            # print("\nLimiar ", linear)

            #Binarização
            binImg = np.zeros((H,W), np.uint8)
            for i in range(H):
                    for j in range(W):
                            if image[i,j] < linear:
                                    binImg[i,j] = 0
                            else:
                                    binImg[i,j] = 255

            jMin = 0
            jMax = 0
            iMin = 0
            iMax = 0

            #Encontrando valor mínimo de j
            for i in range(kd,H):
                for j in range(kd,W):
                    if binImg[i,j] == 0:
                        if jMin==0:
                            jMin = j
                        else:
                            if j < jMin:
                                jMin = j

            #Encontrando valor máximo de j
            for i in range(kd,H):
                for j in range(kd,W):
                    if binImg[i,j] == 0:
                        if jMax==0:
                            jMax = j
                        else:
                            if j > jMax:
                                jMax = j

            #Encontrando valor mínimo de i
            for i in range(kd,H):
                for j in range(kd,W):
                    if binImg[i,j] == 0:
                        if iMin==0:
                            iMin = i
                        else:
                            if i < iMin:
                                iMin = i

            #Encontrando valor máximo de i
            for i in range(kd,H):
                for j in range(kd,W):
                    if binImg[i,j] == 0:
                        if iMax==0:
                            iMax = i
                        else:
                            if i > iMax:
                                iMax = i
            # print(iMin, iMax)
            # print(jMin, jMax)
        
            
            binImgFinal = binImg[(iMin-5):(iMax+5), (jMin-5):(jMax+5)]
            
            return binImgFinal

    #Gerando novas imagens a partir do uso da Binarização por Otsu nas imagens geradas pelos filtros Passa Alta
    binSobel = binaryImage(imSobel,(2*dimS))
    binLaplace = binaryImage(imLaplace,dimL)
    binOriginal = binaryImage(grayImg, 0)

    print("B:",binLaplace.shape)

    iLmin = 0
    iLmax = binLaplace.shape[0]
    jLmin = 0
    jLmax = binLaplace.shape[1]

    lim = 0
    cms = []

    #Cálculo recursivo de centros de massa
    def cMDiv(image, iLmin, iLmax, jLmin, jLmax, lim):
        
        
        acumI = 0
        acumJ = 0
        contP = 0
        
        imgCopia = np.zeros_like(image, dtype=np.float32)
        imgLines = np.zeros_like(image, dtype=np.float32)
        

        

        for i in range(iLmin, iLmax):
            for j in range(jLmin, jLmax):
                imgCopia[i,j] = image[i,j]
                if image[i,j] == 0:
                    contP += 1
                    acumI += i
                    acumJ += j
        
            
        if ((lim<=2) and (contP != 0)):
            cmI = acumI//contP
            cmJ = acumJ//contP
            cms.append((cmI,cmJ))

            lim+=1


            imgLines = cMDiv(imgCopia, iLmin, cmI , jLmin, cmJ,lim)
        
            imgLines = cMDiv(imgCopia, iLmin, cmI , cmJ, jLmax,lim)
            # cv2.imshow("Imagem CM ", imgCopia)
            imgLines = cMDiv(imgCopia, cmI, iLmax , jLmin, cmJ,lim)
            # cv2.imshow("Imagem CM ", imgCopia)
            imgLines = cMDiv(imgCopia, cmI, iLmax , cmJ, jLmax,lim)
            # cv2.imshow("Imagem CM ", imgCopia)

            # imgLines =  cv2.line(imgCopia,(jLmin,cmI),(jLmax,cmI),(0,0,255),1)
            # imgLines =  cv2.line(imgCopia,(cmJ,iLmin),(cmJ, iLmax),(0,0,255),1)

            return imgLines
        else:
            return imgLines

    imCMs = cMDiv(binLaplace, iLmin, iLmax, jLmin, jLmax, lim)

    #Calcular raios na imagem
    def imgRad(x, y, length, angs):
        endx=[]
        endy=[]
        for a in range (len(angs)):
            angle = angs[a]
        
            yf = y + length * math.sin(math.radians(360-angle))
            xf = x + length * math.cos(math.radians(angle))

            endx.append(int(xf))
            endy.append(int(yf))

        return endx, endy

    #Calcula interseções das letras com os raios
    def ptRad(angs,cms, binLaplace):
        # print(binLaplace.shape)
        imgV = np.zeros_like(binLaplace, dtype=np.float32)
        imgH = np.zeros_like(binLaplace, dtype=np.float32)
        for i in range (binLaplace.shape[0]):
            for j in range(binLaplace.shape[1]):
                imgV[i,j] = cms[0][0] - i
                imgH[i,j] = j - cms[0][1]
                
        intersec = []
        # print(imgV)
        # print(imgH)
        pti = (0,binLaplace.shape[1]-cms[0][1]-1) 
        # print(pti)
        npt = np.hypot(pti[0],pti[1])
        # print(npt)       
        for i in range (binLaplace.shape[0]):
            for j in range(binLaplace.shape[1]):
                if binLaplace[i,j]==0:
                    npf = np.hypot(imgV[i,j],imgH[i,j])
                    prI = (pti[0]*imgV[i,j])+(pti[1]*imgH[i,j])
                    angle = math.degrees(math.acos(prI/(npt*npf)))
                    for a in range(len(angs)):
                        if int(angle) == angs[a]:
                            if imgV[i,j]>=0:
                                intersec.append((i,j,angs[a]))
                            else:
                                intersec.append((i,j,360-angs[a]))
                        
        
        return intersec


    # print(len(cms))
    # print(cms)


    imgCMS = np.zeros_like(binLaplace, dtype=np.float32)
    imgR = np.zeros_like(binLaplace, dtype=np.float32)
    for i in range (binLaplace.shape[0]):
        for j in range(binLaplace.shape[1]):
            imgCMS[i,j] = binLaplace[i,j]
            imgR[i,j] = binLaplace[i,j]

    #Desenha pontos nos centros de massa
    for x in range (len(cms)):
        # imgLines =  cv2.line(imgCMS,(0,cms[x][0]),(binLaplace.shape[1],cms[x][0]),(0,0,255),1)
        # imgLines =  cv2.line(imgCMS,(cms[x][1],0),(cms[x][1], binLaplace.shape[0]),(0,0,255),1)
        imgLines =  cv2.line(imgCMS,(cms[x][1],cms[x][0]),(cms[x][1],cms[x][0]),(0,0,255),6)

    #Ângulos dos raios
    angs = [0,30,60,90,120,150,180,210,240,270,300,330]

    endx, endy = imgRad(cms[0][1],cms[0][0], 2*W, angs)

    #Desenha o centro de massa da imagem total
    imgR1 = cv2.line(imgR,(cms[0][1],cms[0][0]),(cms[0][1],cms[0][0]),(0,0,255),6)

    #Desenha os raios na imagem
    for a in range(len(endx)):
        imgR1 = cv2.line(imgR,(cms[0][1],cms[0][0]),(endx[a],endy[a]),(0,0,255),1)

    intersec1 = ptRad(angs,cms, binLaplace)
    # print (intersec1)
    # print (len(intersec1))

    #Desenha pontos nos pixels pretos que fazem interseção com algum dos raios
    for p in range (len(intersec1)):
        imgR1 = cv2.line(imgR,(intersec1[p][1],intersec1[p][0]),(intersec1[p][1],intersec1[p][0]),(0,0,255),6)

    count=0
    intpts = []

    for a in range (len(angs)):
        count=0
        for q in range (len(intersec1)):
            if intersec1[q][2] == angs[a]:
                count += 1
        intpts.append(count)       
            

    print(intpts)

    return image, grayImg, imMedia, imGaussian, imFourier, imSobel, imLaplace, binSobel, binLaplace, binOriginal, imgLines, imgR1, intpts, cms

image1, grayImg1, imMedia1, imGaussian1, imFourier1, imSobel1, imLaplace1, binSobel1, binLaplace1, binOriginal1, imgLines1, imgR11, intpts1, cms1 = proc(image)

image2, grayImg2, imMedia2, imGaussian2, imFourier2, imSobel2, imLaplace2, binSobel2, binLaplace2, binOriginal2, imgLines2, imgR12, intpts2, cms2 = proc(image2)

image3, grayImg3, imMedia3, imGaussian3, imFourier3, imSobel3, imLaplace3, binSobel3, binLaplace3, binOriginal3, imgLines3, imgR13, intpts3, cms3 = proc(image3)


def comparar(binLaplace1, binLaplace2, intpts1, intpts2):
    hMedia = ((binLaplace1.shape[0] + binLaplace2.shape[0])//2)
    wMedia = ((binLaplace1.shape[1] + binLaplace2.shape[1])//2)


    def intpM():
        intpMedia = []
        for v in range (len(intpts1)):
            x = intpts1[v]-intpts2[v]
            intpMedia.append(abs(x))
        return intpMedia

    intpMedia = intpM()

    compatible = 0
    def compativelIP():
        size = hMedia*wMedia
        c = 0
        for v in range (len(intpMedia)):
            if intpMedia[v] > (size/1000):
                c += 1
        if c > 0:
            compatible = False
        else:
            compatible = True
        return compatible

    compatibleIP = compativelIP()

    if compatibleIP == True:
        print("É compatível pelo método das interseções")
    else:
        print("Não é compatível pelo método das interseções")

    def dist(cms):
        for r in range(len(cms)):
            x=0
            x += math.sqrt(((cms[0][0]-cms[r][0])**2)+((cms[0][1]-cms[r][1])**2))
        return x

    dist1 = dist(cms1)
    dist2 = dist(cms2)

    def compativelDist():
        size1 = binLaplace1.shape[0]*binLaplace1.shape[1]
        size2 = binLaplace2.shape[0]*binLaplace2.shape[1]

        v1 = dist1/size1
        v2 = dist2/size2

        if (abs(v1-v2)>0.001):
            compatible = False
        else:
            compatible = True
        return compatible

    compatibleDist = compativelDist()

    if compatibleDist == True:
        print("É compatível pelo método das distâncias")
    else:
        print("Não é compatível pelo método das distâncias")

print("\n Comparação da Imagem 2 com a Imagem 1")
comparar(binLaplace1, binLaplace2, intpts1, intpts2)
print("\n Comparação da Imagem 3 com a Imagem 1")
comparar(binLaplace1, binLaplace3, intpts1, intpts3)

# cv2.imshow("Imagem Colorida", image)  
# cv2.imshow("Imagem Cinza", grayImg1)  
# cv2.imshow("Imagem Filtrada com Media", imMedia1)  
# cv2.imshow("Imagem Filtrada com Gaussiana", imGaussian1)
# cv2.imshow("Imagem Filtrada com Fourier", imFourier1)
# cv2.imshow("Imagem Filtrada com Sobel", imSobel1)
# cv2.imshow("Imagem Filtrada com Laplaciano", imLaplace1)
# cv2.imshow("Imagem Binarizada (Sobel)", binSobel1)
# cv2.imshow("Imagem Binarizada (Laplaciano)", binLaplace1)
# cv2.imshow("Imagem Cinza Original Binarizada ", binOriginal1)
cv2.imshow("Imagem CM ", imgLines1)
cv2.imshow("Imagem CM 2", imgLines2)
cv2.imshow("Imagem CM 3", imgLines3)
cv2.imshow("Imagem com Raios ", imgR11)
cv2.imshow("Imagem com Raios 2 ", imgR12)
cv2.imshow("Imagem com Raios 3 ", imgR13)
cv2.waitKey()   


