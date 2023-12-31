import functions_framework
import cv2
import numpy as np
from google.cloud import storage
from google.cloud import firestore

def descargar_modelo(bucket_name, objeto_nombre, destino):
    """Descarga el modelo YOLO desde GCS."""
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(objeto_nombre)

    blob.download_to_filename(destino)

def cargar_etiquetas(ruta_etiquetas):
    """Carga las etiquetas desde un archivo."""
    with open(ruta_etiquetas, 'r') as file:
        etiquetas = [line.strip() for line in file]
    return etiquetas

def extraer_miniatura(video_path, miniatura_path):
    """Extraer un fotograma del video y guardarlo como miniatura."""
    cap = cv2.VideoCapture(video_path)
    
    # Obtener el primer fotograma
    ret, frame = cap.read()
    
    if ret:
        # Guardar el primer fotograma como miniatura
        cv2.imwrite(miniatura_path, frame)
    
    cap.release()

def subir_miniatura_a_gcs(miniatura_path, bucket_name, miniatura_destino):
    """Subir la miniatura a Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(miniatura_destino)
    
    # Subir la miniatura al bucket
    blob.upload_from_filename(miniatura_path)


def indexar_etiqueta_video(nombre_video, labels, miniatura_url, video_url):
    project_id = 'aprobadasperoaquecosto'
    database_name = 'cloud-base'

    # Crear una instancia de firestore.Client con la base de datos específica
    db = firestore.Client(project=project_id, database=database_name)
    
    # Agregar el video a la colección de "videos" en Firestore
    video_id = nombre_video
    videos_ref = db.collection('videos').document(video_id)
    videos_ref.set({
        'nombre': nombre_video,
        'etiquetas': list(labels),
        'miniatura_url': miniatura_url,
        'video_url': video_url
    }, merge=True)

    # Agregar las etiquetas a la colección de "etiquetas" en Firestore
    for etiqueta in labels:
        etiquetas_ref = db.collection('etiquetas').document(etiqueta)
        etiquetas_ref.set({video_id: True}, merge=True)

def procesar_video(bucket_name, archivo_video):
    """Procesa un video con YOLO."""
    # Descargar el video desde GCS al directorio temporal local
    video_local = f'/tmp/{archivo_video}'
    descargar_modelo(bucket_name, archivo_video, video_local)

    # Descargar archivos de configuración y pesos de YOLO desde GCS
    cfg_local = '/tmp/yolo.cfg'
    weights_local = '/tmp/yolo.weights'
    descargar_modelo('yolov4modellllll', 'yolov4-tiny.cfg', cfg_local)
    descargar_modelo('yolov4modellllll', 'yolov4-tiny.weights', weights_local)

    # Descargar el archivo de etiquetas desde GCS
    ruta_etiquetas = '/tmp/etiquetas.txt'
    descargar_modelo('yolov4modellllll', 'labels.txt', ruta_etiquetas)
    etiquetas = cargar_etiquetas(ruta_etiquetas)

    # Cargar el modelo YOLO y la configuración
    net = cv2.dnn.readNet(weights_local, cfg_local)
    layer_names = net.getUnconnectedOutLayersNames()

    # Crear miniatura
    miniatura_local = '/tmp/miniatura.jpg'
    extraer_miniatura(video_local, miniatura_local)
    subir_miniatura_a_gcs(miniatura_local, 'cloud-imagenes', f'{archivo_video}-thumbnail.jpg')
    url_miniatura = "https://storage.googleapis.com/cloud-imagenes/" + f'{archivo_video}-thumbnail.jpg'


    # Archivo de salida para las detecciones
    salida_txt = '/tmp/salida.txt'
    labels = set()
    with open(salida_txt, 'w') as salida_file:
        # Procesar el video fotograma por fotograma
        cap = cv2.VideoCapture(video_local)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detección de objetos
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(layer_names)

            # Procesar las salidas de la detección y escribir en el archivo de salida
            for out in outs:
                for detection in out:
                    # Obtener la clase con la probabilidad más alta
                    clase_probabilidades = detection[5:]
                    clase_idx = np.argmax(clase_probabilidades)

                    # Obtener la confianza y coordenadas
                    confianza = detection[4]
                    coordenadas = detection[0:4]

                    # Verificar si la confianza es mayor al umbral deseado (por ejemplo, 70%)
                    if confianza > 0.7:
                        # Agregar la clase a labels si la confianza es suficientemente alta
                        if 0 <= clase_idx < len(etiquetas):
                            clase = etiquetas[clase_idx]
                            labels.add(clase)
                            salida_file.write(f'Detección: {clase}, Confianza: {confianza}, Coordenadas: {coordenadas}\n')


        cap.release()

    # Subir el archivo de salida a GCS o almacenar en otro lugar según sea necesario
    storage_client = storage.Client()
    bucket_salida = storage_client.bucket('yolov4modellllll')
    blob_salida = bucket_salida.blob('salida.txt')
    blob_salida.upload_from_filename(salida_txt)

    url_video = "https://storage.googleapis.com/" + bucket_name + "/" + archivo_video
    indexar_etiqueta_video(archivo_video, labels,url_miniatura, url_video)


@functions_framework.cloud_event
def hello_gcs(cloud_event):
    data = cloud_event.data
    """Punto de entrada para la Cloud Function."""
    bucket_name = data['bucket']
    archivo_video = data['name']

    procesar_video(bucket_name, archivo_video)