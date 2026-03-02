import os
import numpy as np
from Segmentation import Masks
from Segmentation.segment_image import segment_with_point, segment_image
from DCM.load_dicom_as_image import read_image_file
import Segmentation.Metrics as Metrics
from Segmentation.negative_points import calculate_negative_point
import cv2

SIMILARITY_THRESHOLD = 0.35  # 30% diferencia aceptable
WARNING_THRESHOLD = 0.45    # 50% diferencia para saltar imagen
DICE_RETRY_THRESHOLD = 0.80  # Reintentar con 2 puntos positivos si Dice < 80%

def resegment_with_mask_center(predictor, image, mask, reference_center):
    """
    Re-segmenta una imagen usando 2 puntos positivos:
    el centro de referencia y el centro calculado de la máscara.
    
    Args:
        predictor: SamPredictor inicializado
        image: Imagen a segmentar (numpy array)
        mask: Máscara de la cual se calcula el centro como segundo punto positivo
        reference_center: Centro de referencia como primer punto positivo
    
    Returns:
        tuple: (new_mask, new_score, mask_center) o (None, None, None) si falla
    """
    mask_center = Masks.calculate_mask_center(mask)
    
    if mask_center is None:
        print(f"    ⚠️ No se pudo calcular centro de la máscara")
        return None, None, None
    
    print(f"    🟢 Punto 1 (centro ref): ({reference_center[0]:.0f}, {reference_center[1]:.0f})")
    print(f"    🟢 Punto 2 (centro máscara): ({mask_center[0]:.0f}, {mask_center[1]:.0f})")
    
    input_points = np.array([reference_center, mask_center])
    input_labels = np.array([1, 1])  # Ambos positivos
    
    img_enhanced = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
    predictor.set_image(img_enhanced)
    
    try:
        new_mask, _, new_score, _ = segment_image(predictor, input_points, input_labels, refine=True)
        
        if new_mask is not None and np.sum(new_mask) > 0:
            return new_mask, new_score, mask_center
        else:
            print(f"    ⚠️ Re-segmentación produjo máscara vacía")
            return None, None, mask_center
    except Exception as e:
        print(f"    ⚠️ Error en re-segmentación con 2 puntos: {e}")
        return None, None, mask_center
    



def propagate_segmentation(predictor, files, start_idx, start_mask, start_center, 
                           segmentations, failed_slices, output_dir, direction="forward"):
    """
    Propaga la segmentación en una dirección (hacia arriba o hacia abajo).
    
    Args:
        predictor: SamPredictor inicializado
        files: Lista de archivos de imágenes
        start_idx: Índice inicial (imagen del medio)
        start_mask: Máscara inicial de referencia
        start_center: Centro inicial de referencia
        segmentations: Diccionario donde guardar resultados (se modifica in-place)
        failed_slices: Lista donde guardar slices fallidas (se modifica in-place)
        output_dir: Directorio de salida
        direction: "forward" (hacia abajo) o "backward" (hacia arriba)
    
    Returns:
        tuple: (segmentations, failed_slices) actualizados
    """
    current_idx = start_idx
    reference_mask = start_mask
    reference_center = start_center
    last_successful_idx = start_idx
    consecutive_failures = 0  # Contador de fallos consecutivos
    MAX_CONSECUTIVE_FAILURES = 3  # Detener propagación tras 2 fallos seguidos
    
    # Configurar dirección
    if direction == "backward":
        step = -1
        condition = lambda idx: idx > 0
        get_next_idx = lambda idx: idx - 1
        emoji = "📤"
        desc = "HACIA ARRIBA (imágenes anteriores)"
    else:  # forward
        step = 1
        condition = lambda idx: idx < len(files) - 1
        get_next_idx = lambda idx: idx + 1
        emoji = "📥"
        desc = "HACIA ABAJO (imágenes posteriores)"
    
    print(f"\n{emoji} PROPAGACIÓN {desc}...")
    
    while condition(current_idx):
        next_idx = get_next_idx(current_idx)
        next_file = files[next_idx]
        next_name = os.path.basename(next_file).split('.')[0]
        
        print(f"\n  [{next_idx+1}/{len(files)}] Procesando {os.path.basename(next_file)}...")
        print(f"    📍 Usando centro de imagen {last_successful_idx+1} (última exitosa)")
        # Verificar si se alcanzó el límite de fallos consecutivos
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"\n    🛑 DETENIENDO propagación {desc}: {consecutive_failures} imágenes consecutivas con diferencia alta.")
            print(f"    Última imagen exitosa: [{last_successful_idx+1}] {os.path.basename(files[last_successful_idx])}")
            break
        
        # Read image
        next_img = read_image_file(next_file)
        if next_img is None:
            print(f"    ❌ Error leyendo imagen. Agregando a fallidas...")
            failed_slices.append({
                'idx': next_idx,
                'filename': os.path.basename(next_file),
                'reason': 'Error leyendo imagen'
            })
            consecutive_failures += 1
            current_idx = next_idx
            continue
        
        # Segment using reference center
        next_mask, next_score = segment_with_point(predictor, next_img, reference_center, label=1, verbose=True)
        
        # Si falla, intentar con offsets
        if next_mask is None or np.sum(next_mask) == 0:
            print(f"    🔄 Intentando con centro ajustado...")
            found_valid = False
            offsets = [(-10, 0), (10, 0), (0, -10), (0, 10), (-20, 0), (20, 0), (0, -20), (0, 20)]
            
            for dx, dy in offsets:
                adjusted_point = [reference_center[0] + dx, reference_center[1] + dy]
                h, w = next_img.shape[:2]
                if 0 <= adjusted_point[0] < w and 0 <= adjusted_point[1] < h:
                    next_mask, next_score = segment_with_point(predictor, next_img, adjusted_point, label=1, verbose=False)
                    if next_mask is not None and np.sum(next_mask) > 0:
                        print(f"    ✅ Encontrada segmentación con offset ({dx}, {dy})")
                        found_valid = True
                        break
            
            if not found_valid:
                print(f"    ❌ Segmentación falló. Agregando a lista de fallidas...")
                failed_slices.append({
                    'idx': next_idx,
                    'filename': os.path.basename(next_file),
                    'reason': 'Segmentación vacía'
                })
                consecutive_failures += 1
                current_idx = next_idx
                continue
        
        # Calculate similarity with reference
        dice = Metrics.calculate_dice_coefficient(reference_mask, next_mask)
        iou = Metrics.calculate_iou(reference_mask, next_mask)
        difference = 1.0 - dice
        
        print(f"    📊 Dice: {dice:.3f}, IoU: {iou:.3f}, Diferencia: {difference*100:.1f}%")
        

        
        # Si la diferencia es mayor al umbral severo, intentar con punto negativo
        if difference > WARNING_THRESHOLD:
            consecutive_failures += 1
            print(f"    🔴 Fallo consecutivo #{consecutive_failures} de {MAX_CONSECUTIVE_FAILURES} permitidos")
            print(f"    ⚠️ Diferencia alta ({difference*100:.1f}%). Intentando con punto negativo...")
            
            # Calcular punto negativo basado en la máscara de referencia
            neg_point = calculate_negative_point(reference_mask, reference_center, distance_factor=0.30)
            
            if neg_point is not None:
                print(f"    🔵 Punto negativo calculado: ({neg_point[0]:.0f}, {neg_point[1]:.0f})")
                
                # Preparar puntos y etiquetas
                input_points = np.array([reference_center, neg_point])
                input_labels = np.array([1, 0])  # 1=positivo, 0=negativo
                
                # Mejorar contraste y segmentar con ambos puntos
                img_enhanced = cv2.convertScaleAbs(next_img, alpha=1.2, beta=10)
                predictor.set_image(img_enhanced)
                
                try:
                    new_mask, _, new_score, _ = segment_image(predictor, input_points, input_labels, refine=True)
                    
                    if new_mask is not None and np.sum(new_mask) > 0:
                        # Calcular nuevo dice
                        new_dice = Metrics.calculate_dice_coefficient(reference_mask, new_mask)
                        new_difference = 1.0 - new_dice
                        
                        print(f"    📊 Con punto negativo - Dice: {new_dice:.3f}, Diferencia: {new_difference*100:.1f}%")
                        
                        # Si mejoró o está dentro del umbral, usar esta segmentación
                        if new_difference <= WARNING_THRESHOLD or new_dice > dice:
                            print(f"    ✅ Punto negativo mejoró la segmentación!")
                            next_mask = new_mask
                            next_score = new_score
                            dice = new_dice
                            iou = Metrics.calculate_iou(reference_mask, new_mask)
                            difference = new_difference
                        else:
                            print(f"    ❌ Punto negativo no mejoró suficiente")
                except Exception as e:
                    print(f"    ⚠️ Error con punto negativo: {e}")
            else:
                print(f"    ⚠️ No se pudo calcular punto negativo")
        
        # Si aún la diferencia es mayor al umbral, SALTAR esta imagen
        if difference > WARNING_THRESHOLD:
            failed_slices.append({
                'idx': next_idx,
                'filename': os.path.basename(next_file),
                'reason': f'Diferencia {difference*100:.1f}% > {WARNING_THRESHOLD*100:.0f}%',
                'dice': dice
            })
            current_idx = next_idx
            continue
        
        # Advertencia leve (pero se acepta)
        if difference > SIMILARITY_THRESHOLD:
            print(f"    ⚠️  Advertencia leve: Diferencia ({difference*100:.1f}%) > {SIMILARITY_THRESHOLD*100:.0f}%")
        
        # Calculate center for next iteration
        next_center = Masks.calculate_mask_center(next_mask)
        
        if next_center is None:
            print(f"    ⚠️  No se pudo calcular centro. Usando centro anterior.")
            next_center = reference_center
        
        print(f"    ✅ EXITOSA. Nuevo centro: ({next_center[0]:.0f}, {next_center[1]:.0f})")
        # Reset contador de fallos consecutivos
        consecutive_failures = 0
        # Guardar resultado
        used_seg_point = list(reference_center)
        segmentations[next_idx] = {
            'mask': next_mask,
            'center': next_center,
            'seg_point': used_seg_point,
            'score': next_score,
            'area': np.sum(next_mask),
            'dice': dice,
            'iou': iou
        }
        
        Masks.save_segmentation_result(next_img, next_mask, next_name, output_dir, 
                                center=next_center, seg_point=used_seg_point, neg_point=None,
                                info=f"Dice: {dice:.3f} | Score: {next_score:.3f}")
        
        # Actualizar referencia
        reference_mask = next_mask
        reference_center = next_center
        last_successful_idx = next_idx
        current_idx = next_idx
    
    return segmentations, failed_slices


def postprocess_low_dice(predictor, files, segmentations, output_dir, dice_threshold=0.80):
    """
    Post-procesamiento: para cada segmentación con dice < umbral,
    va a la imagen anterior, calcula el centro de su máscara,
    y usa ese centro como punto positivo para re-segmentar la imagen anterior.
    
    Ej: imagen 13 tiene dice < 80% → va a imagen 12, calcula centro de máscara 12,
        y re-segmenta imagen 12 con ese centro como punto positivo.
    
    Args:
        predictor: SamPredictor inicializado
        files: Lista de archivos de imágenes
        segmentations: Diccionario de segmentaciones {idx: {mask, center, score, ...}}
        output_dir: Directorio de salida
        dice_threshold: Umbral de dice para re-procesar (default 0.80)
    
    Returns:
        segmentations: Diccionario actualizado
    """
    # Encontrar todas las segmentaciones con dice bajo
    sorted_indices = sorted(segmentations.keys())
    low_dice_indices = [
        idx for idx in sorted_indices
        if 'dice' in segmentations[idx] and segmentations[idx]['dice'] < dice_threshold
    ]
    
    if not low_dice_indices:
        print("✅ No hay segmentaciones con Dice bajo. No se necesita post-procesamiento.")
        return segmentations
    
    print(f"\n🔍 Encontradas {len(low_dice_indices)} segmentaciones con Dice < {dice_threshold*100:.0f}%:")
    for idx in low_dice_indices:
        dice_val = segmentations[idx]['dice']
        print(f"   - [{idx+1}] {os.path.basename(files[idx])}: Dice = {dice_val:.3f}")
    
    resegmented_count = 0
    
    for idx in low_dice_indices:
        # Buscar la imagen anterior que tenga segmentación exitosa
        prev_idx = idx - 1
        while prev_idx >= 0 and prev_idx not in segmentations:
            prev_idx -= 1
        
        if prev_idx < 0 or prev_idx not in segmentations:
            # Si no hay anterior, intentar con la siguiente
            next_idx = idx + 1
            while next_idx < len(files) and next_idx not in segmentations:
                next_idx += 1
            if next_idx >= len(files) or next_idx not in segmentations:
                print(f"\n  ⚠️ [{idx+1}] No hay imagen adyacente con segmentación para re-procesar")
                continue
            prev_idx = next_idx
        
        prev_mask = segmentations[prev_idx]['mask']
        prev_center = Masks.calculate_mask_center(prev_mask)
        
        if prev_center is None:
            print(f"\n  ⚠️ [{idx+1}] No se pudo calcular centro de imagen [{prev_idx+1}]")
            continue
        
        print(f"\n  🔄 [{idx+1}] Dice={segmentations[idx]['dice']:.3f} → Re-segmentando imagen [{prev_idx+1}] con su centro ({prev_center[0]:.0f}, {prev_center[1]:.0f})...")
        
        # Cargar imagen anterior y re-segmentarla con su centro como punto positivo
        prev_img = read_image_file(files[prev_idx])
        if prev_img is None:
            print(f"    ⚠️ No se pudo cargar imagen [{prev_idx+1}]")
            continue
        
        new_mask, new_score, _ = resegment_with_mask_center(
            predictor, prev_img, prev_mask, prev_center
        )
        
        if new_mask is not None:
            new_center = Masks.calculate_mask_center(new_mask)
            
            if new_center is not None:
                # Calcular dice de la nueva máscara vs la original
                new_dice = Metrics.calculate_dice_coefficient(prev_mask, new_mask)
                print(f"    📊 Dice nueva máscara vs original: {new_dice:.3f}")
                
                if new_dice >= dice_threshold:
                    # Actualizar imagen anterior
                    prev_name = os.path.basename(files[prev_idx]).split('.')[0]
                    segmentations[prev_idx]['mask'] = new_mask
                    segmentations[prev_idx]['center'] = new_center
                    segmentations[prev_idx]['score'] = new_score
                    segmentations[prev_idx]['area'] = np.sum(new_mask)
                    
                    Masks.save_segmentation_result(prev_img, new_mask, prev_name, output_dir,
                                            center=new_center, seg_point=list(prev_center),
                                            neg_point=None,
                                            info=f"Post-proc re-seg | Score: {new_score:.3f}")
                    
                    print(f"    ✅ Imagen [{prev_idx+1}] re-segmentada. Nuevo centro: ({new_center[0]:.0f}, {new_center[1]:.0f})")
                    resegmented_count += 1
                else:
                    print(f"    ❌ Dice sigue bajo ({new_dice:.3f} < {dice_threshold}). Manteniendo imagen original [{prev_idx+1}].")
            else:
                print(f"    ⚠️ No se pudo calcular centro de nueva máscara [{prev_idx+1}]")
        else:
            print(f"    ⚠️ Re-segmentación de imagen [{prev_idx+1}] falló")
    
    print(f"\n📊 Post-procesamiento: {resegmented_count}/{len(low_dice_indices)} imágenes anteriores re-segmentadas")
    return segmentations

