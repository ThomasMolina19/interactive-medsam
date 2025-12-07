"""Carga y gestión del modelo SAM."""

import torch
from segment_anything import sam_model_registry, SamPredictor


class SAMModel:
    """Wrapper para el modelo SAM con predicción simplificada."""
    
    def __init__(self, checkpoint_path: str, model_type: str = "vit_b"):
        """
        Inicializa el modelo SAM.
        
        Args:
            checkpoint_path: Ruta al archivo .pth del checkpoint
            model_type: Tipo de modelo SAM ('vit_b', 'vit_l', 'vit_h')
        """
        self.device = self._get_device()
        print(f"\n🖥️  Using device: {self.device}")
        
        print("🔄 Loading SAM model...")
        
        # Cargar checkpoint con map_location para compatibilidad CPU/MPS/CUDA
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # Crear modelo y cargar pesos
        sam = sam_model_registry[model_type]()
        sam.load_state_dict(state_dict)
        sam = sam.to(self.device)
        
        self.predictor = SamPredictor(sam)
        print("✅ SAM model loaded!")
    
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
