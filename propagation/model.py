"""Carga y gestión del modelo SAM."""

import torch
from segment_anything import sam_model_registry, SamPredictor


class SAMModel:
    """Wrapper para el modelo SAM con predicción simplificada."""
    
    def __init__(self, checkpoint_path: str, model_type: str = None, force_cpu: bool = False):
        """
        Inicializa el modelo SAM.
        
        Args:
            checkpoint_path: Ruta al archivo .pth del checkpoint
            model_type: Tipo de modelo SAM ('vit_b', 'vit_l', 'vit_h'). 
                        Si es None, se detecta automáticamente del nombre del archivo.
            force_cpu: Forzar uso de CPU (útil para modelos grandes en MPS)
        """
        # Detectar tipo de modelo automáticamente si no se especifica
        if model_type is None:
            model_type = self._detect_model_type(checkpoint_path)
        
        # Para modelos grandes, usar CPU en Mac para evitar crashes de MPS
        if model_type in ['vit_l', 'vit_h'] and not force_cpu:
            print(f"\n⚠️  Modelo grande ({model_type}) detectado. Usando CPU para estabilidad.")
            self.device = "cpu"
        elif force_cpu:
            self.device = "cpu"
        else:
            self.device = self._get_device()
        
        print(f"🖥️  Using device: {self.device}")
        print(f"🔄 Loading SAM model ({model_type})...")
        
        # Cargar checkpoint con map_location para compatibilidad CPU/MPS/CUDA
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # Crear modelo y cargar pesos
        sam = sam_model_registry[model_type]()
        sam.load_state_dict(state_dict)
        sam = sam.to(self.device)
        
        self.predictor = SamPredictor(sam)
        print("✅ SAM model loaded!")
    
    @staticmethod
    def _detect_model_type(checkpoint_path: str) -> str:
        """Detecta el tipo de modelo SAM desde el nombre del archivo."""
        path_lower = checkpoint_path.lower()
        
        if 'vit_h' in path_lower or 'vith' in path_lower:
            return 'vit_h'
        elif 'vit_l' in path_lower or 'vitl' in path_lower:
            return 'vit_l'
        elif 'vit_b' in path_lower or 'vitb' in path_lower:
            return 'vit_b'
        else:
            # Default a vit_b si no se puede detectar
            print("⚠️  No se pudo detectar tipo de modelo, usando vit_b por defecto")
            return 'vit_b'
    
    @staticmethod
    def _get_device():
        """Determina el mejor dispositivo disponible."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def set_image(self, image):
        """
        Establece la imagen para predicción.
        
        Args:
            image: Array numpy RGB de la imagen
        """
        self.predictor.set_image(image)
    
    def predict(self, point_coords, point_labels, multimask_output=True):
        """
        Ejecuta predicción SAM.
        
        Args:
            point_coords: Coordenadas de los puntos (N, 2)
            point_labels: Labels de los puntos (N,) - 1 positivo, 0 negativo
            multimask_output: Si retorna múltiples máscaras
            
        Returns:
            tuple: (masks, scores, logits)
        """
        return self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output
        )
