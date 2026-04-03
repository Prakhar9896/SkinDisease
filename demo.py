import torch.nn as nn

student_model = nn.Sequential(

    nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(16),
    nn.Hardswish(),

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.Conv2d(16, 24, kernel_size=1),
    nn.BatchNorm2d(24),

    nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1, groups=24),
    nn.BatchNorm2d(24),
    nn.ReLU(),
    nn.Conv2d(24, 40, kernel_size=1),
    nn.BatchNorm2d(40),

    nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1, groups=40),
    nn.BatchNorm2d(40),
    nn.ReLU(),

    nn.Conv2d(40, 160, kernel_size=1),
    nn.BatchNorm2d(160),
    nn.Hardswish(),

    nn.AdaptiveAvgPool2d((1, 1)),

    nn.Flatten(),

    nn.Linear(160, 128),
    nn.Hardswish(),
    nn.Dropout(0.2),
    nn.Linear(128, num_classes)
)