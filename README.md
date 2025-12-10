# Proyecto: Predicción de likes en posts de Instagram

Modelo multimodal en PyTorch que combina imágenes, texto (caption) y variables de metadata para predecir el número de *likes* de un post.

## Estructura del repositorio
- `data/`: carpetas por post (`*.jpg`, `*.txt`, `*.json.xz`).
- `preprocess.py`: genera dataset procesado y escaladores.
- `dataset.py`: dataset PyTorch que carga imágenes (300x300), texto y metadata escalada.
- `models.py`: CNN con *transfer learning* (ResNet18) + MLPs por modalidad + fusión final con salida ReLU.
- `train.py`: entrena el modelo (MSE) guardando el mejor checkpoint.
- `predict.py`: infiere likes para una carpeta de post o todo un directorio.
- `processed/`: artefactos generados (`processed_data.pt`, `vectorizer.joblib`, `text_scaler.joblib`, `meta_scaler.joblib`, `model.pt`).

## Requisitos
```bash
pip install torch torchvision scikit-learn nltk joblib
```

## Preprocesamiento
Limpieza de caption: minúsculas, eliminación de no alfanumérico, tokenizado, stopwords (inglés), lematización. Texto → TF-IDF (`max_features` configurable) → MinMaxScaler. Metadata numérica → MinMaxScaler. Imágenes: se cargan en `dataset.py` y se escalan 0-1 con `ToTensor()` tras `Resize(300,300)`.

Ejecutar:
```bash
python preprocess.py --data_dir data --out_dir processed --max_features 3000
```
Genera `processed/processed_data.pt` y scalers.

## Entrenamiento
- ResNet18 pre-entrenada (congelado todo excepto `layer4`).
- Cabeza de imagen: 512→256→128 ReLU+Dropout.
- Texto: 3000→256→128.
- Metadata: 7→64→32.
- Fusión: concat(128+128+32) → 128 → 1 con ReLU final.
- Pérdida: MSE. Optimizador: Adam. Batch recomendado 1 (cada batch es un post con N imágenes). Device auto (CUDA/MPS/CPU).

Comando:
```bash
python train.py --processed processed/processed_data.pt --epochs 5 --batch_size 1 --lr 1e-4 --val_split 0.2 --model_out processed/model.pt
```

## Predicción
```bash
python predict.py --data_dir data/aashnashroff_969148_3000403601659402518_25980_65 \
  --model processed/model.pt --vectorizer processed/vectorizer.joblib \
  --text_scaler processed/text_scaler.joblib --meta_scaler processed/meta_scaler.joblib
```
También acepta un directorio con múltiples subcarpetas de posts.

## Notas técnicas y decisiones
- Salida ReLU para evitar likes negativos.
- Imágenes de un post se apilan y se promedian en el espacio de features (512d) antes de la cabeza de imagen.
- Se usa `torch.load(..., weights_only=False)` en `dataset.py` por cambio de default en PyTorch 2.6.
- Se fuerza descarga de recursos NLTK `punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`.
- Escalados guardados con `joblib` para reproducibilidad entre entrenamiento y predicción.

## Pasos recomendados
1) Ejecutar `preprocess.py` (una sola vez tras cambios en datos).
2) Entrenar con `train.py` (ajusta épocas/learning rate si hay over/underfit).
3) Probar con `predict.py` en alguna carpeta de post.

## Estado actual
- Preprocesamiento completado (1931 muestras).
- Entrenamiento de smoke-test (1 época) ejecutado; modelo guardado en `processed/model.pt`.
- Predicción funcional; verificar métricas adicionales si se continúa entrenando.
