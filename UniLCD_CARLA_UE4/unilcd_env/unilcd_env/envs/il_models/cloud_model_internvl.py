"""
InternVL3-8B based cloud model for UniLCD.

This module implements a cloud model that uses InternVL3's visual encoder
(InternViT-300M-448px) combined with a custom action head for autonomous driving.
It replaces the original RegNetY-002 based cloud model with a more powerful
vision transformer backbone.

Architecture Overview:
----------------------
1. Visual Encoder: InternViT-300M-448px (extracted from InternVL3-8B)
   - Input: 448x448 RGB images
   - Output: Sequence of visual tokens (256 tokens × hidden_size)

2. Pooling Layer: Converts token sequence to single vector
   - Options: mean pooling, attention pooling, or CLS token

3. Location Embedding: MLP that encodes waypoint information
   - Input: 2D relative waypoint (dx, dy)
   - Output: Vector matching vision hidden size

4. Action Head: MLP that predicts driving actions
   - Input: Concatenated vision + location features
   - Output: [rotation, speed] for vehicle control

Key Design Decisions:
--------------------
- Vision encoder is frozen by default (only action head is trained)
- Optional LoRA fine-tuning for vision encoder adaptation
- Mixed precision (bfloat16) for memory efficiency
- LLM component is deleted to save ~14GB VRAM

Usage:
------
    model = InternVLCloudModel(freeze_vision=True, use_lora=False)
    actions = model(image_tensor, location_tensor)  # Returns [batch, 2]

Author: Generated for UniLCD project
Reference: InternVL3 (https://github.com/OpenGVLab/InternVL)
"""

import torch
import torch.nn as nn
import gc


class InternVLCloudModel(nn.Module):
    """
    InternVL3-8B based cloud model for autonomous driving action prediction.

    This model extracts the visual encoder from InternVL3-8B and adds a custom
    action head to predict steering and speed from camera images.

    Attributes:
        vision_model: The InternViT visual encoder
        vision_hidden_size: Dimensionality of vision features (typically 3584)
        goal: MLP for location/waypoint embedding
        action_head: MLP for action prediction
        pooling_type: Strategy for pooling visual tokens ("mean", "attention", "cls")
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3-8B",
        freeze_vision: bool = True,
        use_lora: bool = False,
        lora_r: int = 16,
        pooling: str = "mean",
        vision_only: bool = True
    ):
        """
        Initialize the InternVL cloud model.

        Args:
            model_name: HuggingFace model ID for InternVL3
                        Default: "OpenGVLab/InternVL3-8B"
            freeze_vision: If True, freeze vision encoder weights (recommended)
                          This means only the action head will be trained
            use_lora: If True, apply LoRA adapters to vision encoder
                     Allows fine-tuning with minimal parameters
            lora_r: LoRA rank (only used if use_lora=True)
                   Higher values = more parameters but better adaptation
            pooling: Strategy for pooling visual tokens to single vector
                    - "mean": Average all tokens (default, most stable)
                    - "attention": Learned attention pooling
                    - "cls": Use CLS token only
            vision_only: If True, only load vision encoder (saves ~14GB VRAM)
        """
        super().__init__()

        self.pooling_type = pooling
        self.vision_only = vision_only

        # Load the InternVL3 model and extract vision encoder
        print(f"Loading InternVL3 model from {model_name}...")
        self._load_vision_encoder(model_name, vision_only)

        # Determine the hidden size of vision features
        # InternVL3-8B uses InternViT with hidden_size=3584
        self.vision_hidden_size = self._get_hidden_size()
        print(f"Vision hidden size: {self.vision_hidden_size}")

        # Freeze vision encoder weights if specified
        # This is recommended for initial training as the vision encoder
        # is already well-trained on diverse visual data
        if freeze_vision and not use_lora:
            print("Freezing vision encoder weights...")
            for param in self.vision_model.parameters():
                param.requires_grad = False

        # Apply LoRA (Low-Rank Adaptation) if specified
        # LoRA allows efficient fine-tuning by adding small trainable matrices
        if use_lora:
            print(f"Applying LoRA with rank={lora_r}...")
            self._apply_lora(lora_r)

        # Attention pooling layer (only used if pooling="attention")
        # Learns to weight different visual tokens based on importance
        if pooling == "attention":
            self.pool_attention = nn.Sequential(
                nn.Linear(self.vision_hidden_size, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )

        # Location/waypoint embedding network
        # Converts 2D waypoint (dx, dy) to high-dimensional feature vector
        # Architecture mirrors the original cloud model's goal embedding
        self.goal = nn.Sequential(
            # First layer: expand 2D input to half of vision hidden size
            nn.Linear(2, self.vision_hidden_size // 2),
            nn.LeakyReLU(negative_slope=0.2),
            # Second layer: expand to full vision hidden size for concatenation
            nn.Linear(self.vision_hidden_size // 2, self.vision_hidden_size)
        )

        # Action prediction head
        # Takes concatenated vision + location features and predicts actions
        # Output: [rotation (steering angle), speed (throttle)]
        self.action_head = nn.Sequential(
            # Input: vision_hidden_size * 2 (vision features + location features)
            nn.Linear(self.vision_hidden_size * 2, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.1),  # Regularization to prevent overfitting
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.1),
            # Output: 2 values (rotation, speed)
            nn.Linear(256, 2)
        )

        # Print parameter statistics
        self._print_param_count()

    def _load_vision_encoder(self, model_name: str, vision_only: bool):
        """
        Load InternVL3 and extract the vision encoder component.

        This method handles the complex process of loading a large VLM
        and extracting only the visual processing components to save memory.

        Args:
            model_name: HuggingFace model identifier
            vision_only: If True, delete the LLM to save memory
        """
        from transformers import AutoModel, AutoConfig

        if vision_only:
            try:
                # Load the full InternVL3 model
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                full_model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
                    low_cpu_mem_usage=True,  # Load weights incrementally
                )

                # Extract the vision model component
                self.vision_model = full_model.vision_model

                # Also extract the MLP projector if it exists
                # This projects vision features to the LLM's input space
                if hasattr(full_model, 'mlp1'):
                    self.vision_projector = full_model.mlp1
                else:
                    self.vision_projector = None

                # Delete the language model to free up ~14GB of VRAM
                # We don't need text generation for action prediction
                if hasattr(full_model, 'language_model'):
                    del full_model.language_model
                del full_model

                # Force garbage collection to reclaim memory
                gc.collect()
                torch.cuda.empty_cache()
                print("Successfully extracted vision encoder, deleted LLM to save memory")

            except Exception as e:
                print(f"Warning: Could not load vision-only: {e}")
                print("Loading full model...")
                self._load_full_model(model_name)
        else:
            self._load_full_model(model_name)

    def _load_full_model(self, model_name: str):
        """
        Load the complete InternVL3 model without memory optimization.

        This is a fallback method used when vision-only loading fails.

        Args:
            model_name: HuggingFace model identifier
        """
        from transformers import AutoModel

        full_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.vision_model = full_model.vision_model

        if hasattr(full_model, 'mlp1'):
            self.vision_projector = full_model.mlp1
        else:
            self.vision_projector = None

    def _get_hidden_size(self) -> int:
        """
        Determine the hidden size of the vision encoder's output.

        This is needed to properly size the downstream MLP layers.
        First tries to get it from config, falls back to a forward pass.

        Returns:
            int: The dimensionality of vision features
        """
        # Try to get from model config first (fastest method)
        if hasattr(self.vision_model, 'config'):
            if hasattr(self.vision_model.config, 'hidden_size'):
                return self.vision_model.config.hidden_size

        # Fallback: run a dummy forward pass to detect output size
        print("Detecting hidden size via forward pass...")
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 448, 448)
            if next(self.vision_model.parameters()).is_cuda:
                dummy_input = dummy_input.cuda()
            dummy_input = dummy_input.to(next(self.vision_model.parameters()).dtype)

            output = self.vision_model(dummy_input)
            if hasattr(output, 'last_hidden_state'):
                hidden_size = output.last_hidden_state.shape[-1]
            else:
                hidden_size = output.shape[-1]

        return hidden_size

    def _apply_lora(self, r: int):
        """
        Apply LoRA (Low-Rank Adaptation) to the vision encoder.

        LoRA adds small trainable matrices to attention layers, allowing
        efficient fine-tuning with minimal additional parameters (~0.1%).

        Args:
            r: LoRA rank - higher values allow more expressive adaptation
               but require more memory. Typical values: 8, 16, 32
        """
        try:
            from peft import LoraConfig, get_peft_model

            # Configure LoRA for attention layers
            lora_config = LoraConfig(
                r=r,  # Rank of the low-rank matrices
                lora_alpha=32,  # Scaling factor
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
                lora_dropout=0.05,  # Dropout for regularization
                bias="none",  # Don't add bias terms
            )
            self.vision_model = get_peft_model(self.vision_model, lora_config)
            print("LoRA applied successfully")
        except ImportError:
            print("Warning: peft not installed, skipping LoRA. Install with: pip install peft")
        except Exception as e:
            print(f"Warning: Could not apply LoRA: {e}")

    def _print_param_count(self):
        """
        Print the number of trainable and total parameters.

        Useful for verifying that freezing/LoRA is working correctly.
        With frozen vision encoder, only ~1M parameters should be trainable.
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    def pool_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Pool visual token sequence to a single feature vector.

        The vision encoder outputs a sequence of tokens (one per image patch).
        This method aggregates them into a single vector for the action head.

        Args:
            hidden_states: Vision tokens [batch_size, num_tokens, hidden_size]
                          Typically num_tokens = 256 for 448x448 images

        Returns:
            Pooled features [batch_size, hidden_size]
        """
        if self.pooling_type == "mean":
            # Simple average over all tokens - most stable option
            return hidden_states.mean(dim=1)
        elif self.pooling_type == "cls":
            # Use only the first token (CLS token)
            # May lose spatial information
            return hidden_states[:, 0]
        elif self.pooling_type == "attention":
            # Learned attention pooling - weights tokens by importance
            # More expressive but requires more training
            attn_weights = self.pool_attention(hidden_states)  # [B, N, 1]
            attn_weights = torch.softmax(attn_weights, dim=1)  # Normalize
            return (hidden_states * attn_weights).sum(dim=1)  # Weighted sum
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

    def forward(self, pixel_values: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict driving actions from image and waypoint.

        This is the main inference method. Given a camera image and the
        relative position of the next waypoint, it predicts the steering
        angle and speed for the vehicle.

        Args:
            pixel_values: Normalized image tensor [batch, 3, 448, 448]
                         Should be normalized with ImageNet mean/std
            locations: Relative waypoint position [batch, 2]
                      Format: (dx, dy) = current_pos - next_waypoint

        Returns:
            actions: Predicted driving actions [batch, 2]
                    Format: [rotation (radians), speed (normalized)]

        Example:
            >>> model = InternVLCloudModel()
            >>> image = preprocess(camera_frame)  # [1, 3, 448, 448]
            >>> waypoint = torch.tensor([[dx, dy]])  # [1, 2]
            >>> steering, speed = model(image, waypoint)[0]
        """
        # Ensure inputs are on the correct device
        device = next(self.parameters()).device
        pixel_values = pixel_values.to(device)
        locations = locations.to(device)

        # Extract visual features using InternViT
        # Use mixed precision for memory efficiency
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            vision_outputs = self.vision_model(pixel_values)

            # Extract the hidden states (token sequence)
            if hasattr(vision_outputs, 'last_hidden_state'):
                hidden_states = vision_outputs.last_hidden_state
            else:
                hidden_states = vision_outputs

        # Pool tokens to single vector (convert to float32 for stability)
        vision_features = self.pool_features(hidden_states.float())  # [B, hidden_size]

        # Encode the waypoint location
        location_features = self.goal(locations.float())  # [B, hidden_size]

        # Concatenate vision and location features
        combined = torch.cat([vision_features, location_features], dim=1)

        # Predict driving actions
        actions = self.action_head(combined)

        return actions


class InternVLCloudModelLite(nn.Module):
    """
    Lightweight version using InternViT-300M encoder directly.

    This is a simpler alternative that loads only the standalone InternViT
    model without the full InternVL3 framework. Use this if:
    - You have limited VRAM (<16GB)
    - You want faster loading/inference
    - You don't need the full InternVL3 capabilities

    The architecture is identical to InternVLCloudModel but uses a smaller
    vision encoder (~300M parameters instead of InternVL3's encoder).
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternViT-300M-448px",
        freeze_vision: bool = True,
        pooling: str = "mean"
    ):
        """
        Initialize the lite version with standalone InternViT.

        Args:
            model_name: HuggingFace model ID for InternViT
            freeze_vision: If True, freeze vision encoder weights
            pooling: Token pooling strategy ("mean", "cls", or "attention")
        """
        super().__init__()

        from transformers import AutoModel

        print(f"Loading InternViT from {model_name}...")
        self.vision_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Lite version uses fp32
        )

        # Get hidden size from config
        if hasattr(self.vision_model.config, 'hidden_size'):
            self.vision_hidden_size = self.vision_model.config.hidden_size
        else:
            self.vision_hidden_size = 1024  # Default for InternViT-300M

        print(f"Vision hidden size: {self.vision_hidden_size}")

        # Freeze vision encoder if specified
        if freeze_vision:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        self.pooling_type = pooling

        # Location embedding MLP
        self.goal = nn.Sequential(
            nn.Linear(2, self.vision_hidden_size // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.vision_hidden_size // 2, self.vision_hidden_size)
        )

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(self.vision_hidden_size * 2, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 2)
        )

    def forward(self, pixel_values: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the lite model.

        Args:
            pixel_values: Image tensor [batch, 3, 448, 448]
            locations: Waypoint tensor [batch, 2]

        Returns:
            actions: Predicted [rotation, speed] tensor [batch, 2]
        """
        device = next(self.parameters()).device
        pixel_values = pixel_values.to(device)
        locations = locations.to(device)

        # Get vision features
        vision_outputs = self.vision_model(pixel_values)
        if hasattr(vision_outputs, 'last_hidden_state'):
            hidden_states = vision_outputs.last_hidden_state
        else:
            hidden_states = vision_outputs

        # Mean pooling over tokens
        vision_features = hidden_states.mean(dim=1)

        # Location embedding
        location_features = self.goal(locations)

        # Predict actions
        combined = torch.cat([vision_features, location_features], dim=1)
        actions = self.action_head(combined)

        return actions


# =============================================================================
# Testing and Verification
# =============================================================================
if __name__ == "__main__":
    """
    Test script to verify model loading and forward pass.

    Run with: python cloud_model_internvl.py
    """
    print("=" * 60)
    print("Testing InternVLCloudModel")
    print("=" * 60)

    # Initialize model with frozen vision encoder
    model = InternVLCloudModel(
        freeze_vision=True,
        use_lora=False,
        pooling="mean"
    )
    model = model.cuda().eval()

    # Create dummy inputs
    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 448, 448).cuda()
    locations = torch.randn(batch_size, 2).cuda()

    # Run forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        actions = model(pixel_values, locations)

    # Verify output
    print(f"\nInput shapes:")
    print(f"  - pixel_values: {pixel_values.shape}")
    print(f"  - locations: {locations.shape}")
    print(f"\nOutput shape: {actions.shape}")
    print(f"Output values: {actions}")

    # Check output range
    print(f"\nOutput statistics:")
    print(f"  - Rotation range: [{actions[:, 0].min():.3f}, {actions[:, 0].max():.3f}]")
    print(f"  - Speed range: [{actions[:, 1].min():.3f}, {actions[:, 1].max():.3f}]")

    print("\n" + "=" * 60)
    print("Test PASSED!")
    print("=" * 60)
