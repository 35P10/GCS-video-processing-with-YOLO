import functions_framework
import cv2
import numpy as np
from google.cloud import storage
from google.cloud import firestore

# Constantes de configuración
BUCKET_MODELOS = 'yolov4modellllll'
CFG_YOLO = 'yolov4-tiny.cfg'
WEIGHTS_YOLO = 'yolov4-tiny.weights'
LABELS_FILE = 'labels.txt'
BUCKET_IMAGENES = 'cloud-imagenes'
PROJECT_ID = 'aprobadasperoaquecosto'
DATABASE_NAME = 'cloud-base'

def descargar_modelo(bucket_name, objeto_nombre, destino):
    """Descarga el modelo YOLO desde GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(objeto_nombre)
    blob.download_to_filename(destino)

def cargar_etiquetas(ruta_etiquetas):
    """Carga las etiquetas desde un archivo."""
    with open(ruta_etiquetas, 'r') as file:
        return [line.strip() for line in file]

def extraer_miniatura(video_path, miniatura_path):
    """Extraer un fotograma del video y guardarlo como miniatura."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(miniatura_path, frame)
    cap.release()

def subir_miniatura_a_gcs(miniatura_path, bucket_name, miniatura_destino):
    """Subir la miniatura a Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(miniatura_destino)
    blob.upload_from_filename(miniatura_path)

def indexar_etiqueta_video(nombre_video, labels, miniatura_url, video_url):
    """Agregar información del video a Firestore."""
    db = firestore.Client(project=PROJECT_ID, database=DATABASE_NAME)
    video_id = nombre_video
    videos_ref = db.collection('videos').document(video_id)
    videos_ref.set({
        'nombre': nombre_video,
        'etiquetas': list(labels),
        'miniatura_url': miniatura_url,
        'video_url': video_url
    }, merge=True)
    for etiqueta in labels:
        etiquetas_ref = db.collection('etiquetas').document(etiqueta)
        etiquetas_ref.set({video_id: True}, merge=True)

def procesar_video(bucket_name, archivo_video):
    """Procesa un video con YOLO."""
    video_local = f'/tmp/{archivo_video}'
    descargar_modelo(bucket_name, archivo_video, video_local)

    cfg_local = f'/tmp/{CFG_YOLO}'
    weights_local = f'/tmp/{WEIGHTS_YOLO}'
    ruta_etiquetas = f'/tmp/{LABELS_FILE}'

    # Descargar archivos de configuración y etiquetas
    descargar_modelo(BUCKET_MODELOS, CFG_YOLO, cfg_local)
    descargar_modelo(BUCKET_MODELOS, WEIGHTS_YOLO, weights_local)
    descargar_modelo(BUCKET_MODELOS, LABELS_FILE, ruta_etiquetas)

    etiquetas = cargar_etiquetas(ruta_etiquetas)
    net = cv2.dnn.readNet(weights_local, cfg_local)
    layer_names = net.getUnconnectedOutLayersNames()

    miniatura_local = f'/tmp/miniatura.jpg'
    extraer_miniatura(video_local, miniatura_local)
    subir_miniatura_a_gcs(miniatura_local, BUCKET_IMAGENES, f'{archivo_video}-thumbnail.jpg')
    url_miniatura = f"https://storage.googleapis.com/{BUCKET_IMAGENES}/{archivo_video}-thumbnail.jpg"

    labels = set()
    with cv2.VideoCapture(video_local) as cap:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(layer_names)

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    clase_idx = np.argmax(scores)
                    confianza = detection[4]

                    if confianza > 0.7 and 0 <= clase_idx < len(etiquetas):
                        clase = etiquetas[clase_idx]
                        labels.add(clase)

    url_video = f"https://storage.googleapis.com/{bucket_name}/{archivo_video}"
    indexar_etiqueta_video(archivo_video, labels, url_miniatura, url_video)

@functions_framework.cloud_event
def hello_gcs(cloud_event):
    """Punto de entrada para la Cloud Function."""
    data = cloud_event.data
    bucket_name = data['bucket']
    archivo_video = data['name']
    procesar_video(bucket_name, archivo_video)