import read_bfee
import get_scaled_csi
import socket               # Importar el módulo de socket
import numpy as np
import pyqtgraph as pg
import threading
import queue

# Clase que contiene amplitud y fase de una lectura CSIisRunning
class CSIData:
    def __init__(self, scaledCSI):
        self.scaledCSI = scaledCSI

#Hilo para procesar la informacion recibida por el puerto de comunicacion serial
def processThread ():
    global clientConnection
    while True:
        msg = msgQueue.get(block = True) #bloqueara el hilo hasta que haya un elemento disponible
        scaledCSI = []  # Matriz que ira almacenando los valores escalados de CSI
        try:
            fl = msg.read(2)
            field_len = int.from_bytes(fl, 'big')
            if field_len == 0:
                clientConnection = False
                break
        except: # se podría establecer una variable bool, que indique que el programa ha perdido conexión con el cliente.
             ## con una variable global
            print('Timeout, please restart the client and connect again.')
            break
        co = msg.read(1)
        if not isinstance(co, int):
            code = int.from_bytes(co, 'big')
        else:
            code = co

        # If unhandled code, skip (seek over) the record and continue

        if code == 187:  # get beamforming or phy data
            Bytes = msg.read(field_len - 1)
            if len(Bytes) != field_len - 1:
                c.close()
                exit()

        elif field_len <= 1024:  # skip all other info
            msg.read(field_len - 1)
            continue
        else:
            continue

        if code == 187:
            csi_entry = read_bfee.read_bfee(Bytes)
            if not csi_entry:
                print('Error: malformed packet')
                exit()

            perm = csi_entry[9]
            Nrx = csi_entry[2]
            if Nrx > 1:
                if sum(perm) != triangle[Nrx - 1]:
                    if broken_perm == 0:
                        broken_perm = 1
                        print('WARN ONCE: Found CSI with Nrx=', Nrx, ' and invalid perm=\n')
                    else:
                        csi_entry[11][:, perm[0:Nrx], :] = csi_entry[11][:, 0:Nrx, :]

            csi = get_scaled_csi.get_scaled_csi(csi_entry)

        #Se obtiene un arreglo de la parte real e imaginaria de CSI
        for i in range(len(csi)): #For para cada elemeno en csi
            scaledCSI.append(csi[i])

        newCSI = CSIData(scaledCSI)
        rdyQueue.put(newCSI)


#Hilo para escribir en archivo

def plotThread():
    while True:
        try:
            csiDatos = rdyQueue.get(block=True)

            if csiDatos == 0:
                break
        except:
            break

        x = np.arange(30)
        #  p1.setData(x, db(abs(frames[0].csi_matrix[0][0])).transpose().flatten())))
        #  p2.setData(x, db(abs(frames[0].csi_matrix[0][1])).transpose().flatten())))
        #  p3.setData(x, db(abs(frames[0].csi_matrix[0][2])).transpose().flatten())))
        p1.setData(x, get_scaled_csi.db(abs(csiDatos.scaledCSI[0][0][:].flatten()))) 
        p2.setData(x, get_scaled_csi.db(abs(csiDatos.scaledCSI[0][1][:].flatten())))
        p3.setData(x, get_scaled_csi.db(abs(csiDatos.scaledCSI[0][2][:].flatten())))


def writingThread():
    while True:
        try:
            csiDatos = rdyQueue.get(block=True)

            if csiDatos == 0:
                break
        except:
            break

        csi_1.append(csiDatos.scaledCSI[0][0][:])
        csi_2.append(csiDatos.scaledCSI[0][1][:])
        csi_3.append(csiDatos.scaledCSI[0][2][:])
        #csi_1.append(get_scaled_csi.db(abs(np.squeeze(csiDatos.scaledCSI[0][0][:]))))
        #csi_2.append(get_scaled_csi.db(abs(np.squeeze(csiDatos.scaledCSI[0][1][:]))))
        #csi_3.append(get_scaled_csi.db(abs(np.squeeze(csiDatos.scaledCSI[0][2][:]))))


csi_1 = []
csi_2 = []
csi_3 = []
clientConnection = True  ## Indica si existe actualmente conexión con el cliente
print("Inicio")
#threads.join - Para que el hilo principal espere a que terminen los hilos hijos
while True:
    #FIFO que almacena de manera temporal los mensajes recibidos, tamaño indefinido...
    msgQueue = queue.Queue()
    #FIFO que almacena el mensaje procesado listo para ser escrito en archivo
    rdyQueue = queue.Queue()

    win = pg.GraphicsWindow()
    p = win.addPlot()

    p.setLabel('left', "SNR", units='dB')  
    p.setLabel('bottom', "subcarrier index") 
    p.setRange(xRange = [0, 30], yRange=[-150, 0])

    p1 = p.plot(pen = 'r')
    p2 = p.plot(pen = 'y')
    p3 = p.plot(pen = 'b')

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)         # Crear objeto de socket

    host = socket.gethostname() # Obtener el nombre de localhost
    port = 8090                # Establecer puerto
    s.bind((host, port))        # Puerto de enlace
    s.listen(5)                 # En espera que el cliente se conecte
    print('waiting for connection on port', port)

    c, addr = s.accept()     # Establecer conexión con el cliente
    c.settimeout(15)
    #buffersize = 1024  # 204800: No se grafica; 102400 (a veces no se grafica), 20480 y 1024: Se grafica y solo dura 4 segs
    #c.recv(1024) # Añadido por Emmanuel López Hernández

    print('Dirección de conexión：', addr)
    fd = c.makefile('rb')

    #while True:

    csi_entry = []
    index = -1                     # The index of the plots which need shadowing
    broken_perm = 0                # Flag marking whether we've encountered a broken CSI yet
    triangle = [1,3,6]             # What perm should sum to for 1,2,3 antennas
    current_index = 0
    p = np.zeros((30,30), dtype='double', order = 'C')

    # Se activan los threads
    pthread = threading.Thread(target=processThread, daemon=True)
    pthread.start()
    gthread = threading.Thread(target=plotThread, daemon=True)
    gthread.start()
    wthread = threading.Thread(target=writingThread, daemon=True)
    wthread.start()


    while clientConnection == True: # poner condición de salida
        # Read size and code from the received packets
        msgQueue.put(fd)
        pg.QtGui.QApplication.processEvents()

    c.close()                # Cerrar la conexión - No está llegando aquí, se corta
    pthread.join()
    gthread.join()
    wthread.join()

    pg.QtGui.QApplication.closeAllWindows()
    print("Final")
    CsvNewFile = np.zeros([len(np.transpose(csi_1)) + 1, len(np.transpose(csi_2)) + 1, len(np.transpose(csi_3)) + 1])
    CsvNewFile = np.hstack([csi_1, csi_2, csi_3])
    DFCsv = pd.DataFrame(CsvNewFile)
    DFCsv.to_csv(r'' + 'datos_crudos' + '/csi_sin_escalar.csv', index=False)
    break

    """
    csi_amp_matrix = np.zeros([len(amp_1), 90])
    csi_amp_matrix[:, 0:30] = amp_1
    csi_amp_matrix[:, 30:60] = amp_2
    csi_amp_matrix[:, 60:90] = amp_3
    """
