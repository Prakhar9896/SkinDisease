import torch
import torch.nn as nn
import timm

class EfficientNetB2Unified(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Note: pretrained=False since we are loading our own weights anyway
        self.backbone = timm.create_model('efficientnet_b2', pretrained=False, num_classes=0, global_pool='')
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(1408),
            nn.Dropout(0.3),
            nn.Linear(1408, num_classes),
        )
    def forward(self, x):
        return self.head(self.backbone.forward_features(x))

def export_to_onnx():
    num_classes = 16 
    model = EfficientNetB2Unified(num_classes)
    
    ckpt_path = 'checkpoints/stage3_replay_fitz_tuned.pth'
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    if 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)
        
    model.eval()
    
    # EfficientNet-B2 expects 260x260 input
    dummy_input = torch.randn(1, 3, 260, 260, requires_grad=True)
    
    onnx_path = "model.onnx"
    print(f"Exporting model to {onnx_path}...")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Export successful! File saved at: {onnx_path}")

if __name__ == '__main__':
    export_to_onnx()
