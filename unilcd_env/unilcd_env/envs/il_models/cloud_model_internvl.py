"""
InternVL3-2B Cloud Model for UniLCD
Replaces RegNetY-002 with InternVL3-2B vision-language model for improved scene understanding.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torchvision.transforms as transforms


class CloudModelInternVL(nn.Module):
    """
    Cloud model using InternVL3-2B for autonomous driving action prediction.

    Input:
        - x: Image tensor [B, 3, H, W] (normalized)
        - locations: Waypoint offset tensor [B, 2] (delta_x, delta_y)

    Output:
        - actions: [B, 2] (steering_angle, speed)
    """

    def __init__(self, model_name="OpenGVLab/InternVL3-2B", device=None):
        super(CloudModelInternVL, self).__init__()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load InternVL model
        print(f"Loading InternVL model: {model_name}")
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Freeze the base model for inference (unfreeze for fine-tuning)
        for param in self.model.parameters():
            param.requires_grad = False

        # Get hidden size from model config
        self.hidden_size = self.model.config.llm_config.hidden_size

        # Goal/location embedding network
        self.goal = nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 512)
        )

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_size + 512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 2)  # steering, speed
        )

        # Image preprocessing for InternVL
        self.preprocess = transforms.Compose([
            transforms.Resize((448, 448)),  # InternVL expects 448x448
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"InternVL CloudModel initialized. Hidden size: {self.hidden_size}")

    def get_image_features(self, pixel_values):
        """Extract visual features from InternVL's vision encoder."""
        # Use the vision model to get image features
        vision_outputs = self.model.vision_model(pixel_values)
        image_features = vision_outputs.last_hidden_state

        # Pool to get a single feature vector per image
        # Use mean pooling over spatial dimensions
        image_features = image_features.mean(dim=1)  # [B, hidden_size]

        return image_features

    def forward(self, x, locations):
        """
        Forward pass for action prediction.

        Args:
            x: Image tensor [B, 3, H, W] - already normalized
            locations: Location tensor [B, 2] - (delta_x, delta_y) to next waypoint

        Returns:
            actions: [B, 2] - (steering_angle, speed)
        """
        # Resize images to InternVL's expected size
        if x.shape[-2:] != (448, 448):
            x = nn.functional.interpolate(x, size=(448, 448), mode='bilinear', align_corners=False)

        # Convert to bfloat16 if model is in bfloat16 (check via list to avoid StopIteration with DataParallel)
        params = list(self.model.parameters())
        if params and params[0].dtype == torch.bfloat16:
            x = x.to(torch.bfloat16)
            locations = locations.to(torch.bfloat16)

        # Get image features from vision encoder (no torch.no_grad() to allow LoRA training)
        image_features = self.get_image_features(x)

        # Get goal embeddings
        goal_features = self.goal(locations)

        # Concatenate image and goal features
        combined = torch.cat([image_features.float(), goal_features], dim=1)

        # Predict actions
        actions = self.action_head(combined)

        return actions

    def forward_with_prompt(self, x, locations, prompt="Predict steering and speed for autonomous driving."):
        """
        Alternative forward pass using VLM's full capabilities with text prompt.
        This is slower but may provide better scene understanding.

        Note: This requires the full InternVL pipeline and is not used by default.
        """
        # This method is for experimentation - uses text prompts
        # For real-time inference, use the standard forward() method
        raise NotImplementedError("Prompt-based inference not implemented for real-time use")


class CloudModelInternVLLite(nn.Module):
    """
    Lightweight version that only uses InternVL's vision encoder.
    Faster inference, suitable for real-time applications.
    """

    def __init__(self, model_name="OpenGVLab/InternVL3-2B", device=None):
        super(CloudModelInternVLLite, self).__init__()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading InternVL vision encoder: {model_name}")

        # Load only the vision model part
        full_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # Extract vision model
        self.vision_model = full_model.vision_model
        self.hidden_size = full_model.config.vision_config.hidden_size

        # Delete the LLM part to save memory
        del full_model
        torch.cuda.empty_cache()

        # Freeze vision encoder
        for param in self.vision_model.parameters():
            param.requires_grad = False

        # Goal embedding
        self.goal = nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 512)
        )

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_size + 512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 2)
        )

        print(f"InternVL Lite CloudModel initialized. Vision hidden size: {self.hidden_size}")

    def forward(self, x, locations):
        """Forward pass using only vision encoder."""
        # Resize to expected size
        if x.shape[-2:] != (448, 448):
            x = nn.functional.interpolate(x, size=(448, 448), mode='bilinear', align_corners=False)

        # Always convert to bfloat16 (InternVL3 uses bfloat16)
        x = x.to(torch.bfloat16)
        locations = locations.to(torch.bfloat16)

        # Get vision features (no torch.no_grad() to allow LoRA training)
        vision_out = self.vision_model(x)
        image_features = vision_out.last_hidden_state.mean(dim=1)

        # Goal features
        goal_features = self.goal(locations.float())

        # Combine and predict
        combined = torch.cat([image_features.float(), goal_features], dim=1)
        actions = self.action_head(combined)

        return actions
