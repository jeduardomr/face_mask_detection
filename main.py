import os
import sys
from pathlib import Path
from utils.config import Config

def main():
    """
    FUNCIÓN PRINCIPAL DEL SISTEMA - VERSIÓN YOLO
    """
    print("=" * 60)
    print("SISTEMA DE DETECCION DE MASCARILLAS EN TIEMPO REAL (YOLO11n)")
    print("=" * 60)
    
    # 1. CREAR DIRECTORIOS NECESARIOS
    Config.create_directories()
    
    # 2. VERIFICACIÓN RÁPIDA DEL DATASET (sin depender de verificar_dataset.py)
    images_dir = Path("data/face-mask-data/images")
    labels_dir = Path("data/face-mask-data/labels")
    
    if not images_dir.exists() or not labels_dir.exists():
        print("❌ ERROR: No se encuentran las carpetas 'images' o 'labels' en data/face-mask-data/")
        print("   Asegúrate de tener el dataset en la estructura correcta.")
        return
    
    num_images = len(list(images_dir.glob('*.*')))
    num_labels = len(list(labels_dir.glob('*.txt')))
    
    if num_images == 0 or num_labels == 0:
        print("❌ ERROR: Las carpetas 'images' o 'labels' están vacías.")
        print(f"   Imágenes encontradas: {num_images}")
        print(f"   Etiquetas encontradas: {num_labels}")
        return
    
    print(f"✅ Dataset verificado: {num_images} imágenes, {num_labels} etiquetas.")
    
    # ========== RUTA DEL MODELO YOLO ENTRENADO ==========
    yolo_model_path = Path("runs/detect/yolo11n_facemask/weights/best.pt")
    
    # 3. ENTRENAR O CARGAR MODELO YOLO
    if not yolo_model_path.exists():
        print("🔄 Modelo YOLO no encontrado. Iniciando entrenamiento...")
        print("⚠️  Este proceso tomará entre 30 min y 2 horas (dependiendo de tu GPU).")
        print("   El modelo se guardará automáticamente cuando termine.\n")
        
        try:
            from training.train_yolo import train_yolo
            train_yolo()  # ← Ejecuta el entrenamiento completo
            print("✅ Entrenamiento completado exitosamente.")
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {e}")
            return
    else:
        print("✅ Modelo YOLO pre-entrenado encontrado!")
    
    # 4. INICIAR DETECCIÓN EN TIEMPO REAL CON YOLO
    print("\n🎥 Iniciando detección en tiempo real...")
    print("-" * 50)
    
    try:
        from detection.real_time_detection import RealTimeMaskDetection
        detector = RealTimeMaskDetection(str(yolo_model_path))
        detector.run_detection()
    except Exception as e:
        print(f"❌ Error durante la detección: {e}")
        print("   Asegúrate de que la cámara esté conectada y disponible.")
        print("   Si el error persiste, revisa la instalación de OpenCV y pygame.")

if __name__ == "__main__":
    main()