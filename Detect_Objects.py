import sys
import dlib
import cv2
import time
from datetime import datetime
import pandas as pd
import warnings
import glob
warnings.filterwarnings("ignore")

def Percent(value):
    if value >= 1.0:
        return 100
    else:
        x = str('{:.0%}'.format(value))

        return int(x.split('%')[0])


#INICIALIZAÇÃO DE LISTAS DE LOGS
timestamp = []
produtoNomeLog = []
assertividade = []

pula_quadros = 1
captura = cv2.VideoCapture(0)
contadorQuadros =  0

font = cv2.cv2.FONT_HERSHEY_DUPLEX

#============== PARAMETERS ==========================
#SEGUNDOS PARA EXIBIR O QRCODE
segundosExbicao = 5
taxaDeErro = 50 #Capture objects above this percent
resolucao = 1
#====================================================

path = 'SVMs Processed\\'
pathSvms = glob.glob(path + '*')

qrCodeNome = []
qrCodeImages = []

#LOADING SVMS
produtosTreinados = []
nomeProdutosTreinados = []
for svm in pathSvms:  
    produtosTreinados.append(dlib.fhog_object_detector(svm))
    nomeProdutosTreinados.append(svm.split('-')[1].replace('.svm',''))
    
    qrCodeImages.append(cv2.imread('QR\\qr-code-{produto}.png'.format(produto = svm.split('-')[1].replace('.svm',''))))
    qrCodeNome.append(svm.split('-')[1].replace('.svm',''))

#Resize Imgs
ind = 0
for x in qrCodeImages:
    qrCodeImages[ind] = cv2.resize(qrCodeImages[ind],(100,100))
    ind+=1

while captura.isOpened():
    conectado, frame = captura.read()
    [boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(produtosTreinados, frame, upsample_num_times=1, adjust_threshold=0.0)

    #TRATATIVA PARA N LER TODOS OS FRAMES
    contadorQuadros += 1
    if contadorQuadros % pula_quadros == 0:

        index_nome = 0
        for o in boxes:
            e, t, d, f = (int(o.left()), int(o.top()), int(o.right()), int(o.bottom()))
            
            if Percent(confidences[index_nome]) >= taxaDeErro:

                #SQUARE
                cv2.rectangle(frame, (e, t), (d, f), (0, 0, 255), 2)

                #PRODUCT NAME
                cv2.putText(frame, nomeProdutosTreinados[detector_idxs[index_nome]], (e,f +30),  font,   1.0,    (0, 0, 255),  2)

                #CONFIDENCE 
                cv2.putText(frame, str(Percent(confidences[index_nome])) + '%', (e,t-10),  font,   1.0,    (0, 0, 255),  2)


                #SHOW QRCode
                frame[10:110,10:110] = qrCodeImages[detector_idxs[index_nome]]
                cv2.putText(frame,'Visite o site para mais informacoes',(120, 70),font,.65,(255,255,255),2)

                #Logs
                timestamp.append(datetime.now())
                assertividade.append(str(Percent(confidences[index_nome])) + '%')
                produtoNomeLog.append(nomeProdutosTreinados[detector_idxs[index_nome]])

            index_nome+=1

    cv2.imshow("Preditor de Objetos", frame) #ESC TO EXIT
    if cv2.waitKey(1) & 0xFF == 27:
        break


df = pd.DataFrame(data = {'NomeProduto':produtoNomeLog,'Assertividade':assertividade,'Timestamp':timestamp})
path = 'Logs\\'
df.to_excel(path+'Log_{day}_{month}_{year}_{hour}_{min}_{secs}.xlsx'.format(day = datetime.now().day,
                                                                            month = str(datetime.now().month),
                                                                            year = str(datetime.now().year),
                                                                            hour = str(datetime.now().hour),
                                                                            min = str(datetime.now().minute),
                                                                            secs = str(datetime.now().second)),index = False)


captura.release()
cv2.destroyAllWindows()
sys.exit(0)