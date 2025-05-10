import cv2
import numpy as np
from pathlib import Path
import infiray_irg # Library for parsing .irg thermal image files
import matplotlib.pyplot as plt
from tqdm import tqdm # For progress bar
import sys
import traceback
import json

def apply_colormap(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply colormap using Lookup Table (LUT) with per-channel processing.
    
    Args:
        image: Input thermal image (single-channel or multi-channel)
        lut: Plasma colormap Lookup Table (256x3 uint8 array)
    
    Returns:
        Color-mapped image (BGR format) or black image on error
    """
    try:
        # Validate input data
        if image is None or image.size == 0:
            raise ValueError("Empty input image")

        # Normalize and convert to uint8 if needed
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Ensure 3-channel format (H, W, 3)
        if len(image.shape) == 2:  # Single-channel (grayscale)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:  # Single-channel in multi-channel array
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Validate and prepare LUT
        if lut.shape != (256, 3) or lut.dtype != np.uint8:
            raise ValueError(f"LUT must be (256, 3) uint8, got {lut.shape} {lut.dtype}")
        lut = np.ascontiguousarray(lut)  # Improve LUT access speed

        # Apply LUT to each channel separately (BGR order)
        channels = []
        for i in range(3):
            channel = image[:, :, i]
            channel_mapped = cv2.LUT(channel, lut[:, i])
            channels.append(channel_mapped)

        return np.stack(channels, axis=2)

    except Exception as e:
        print(f"Error in apply_plasma: {str(e)}")
        return np.zeros((100, 100, 3), dtype=np.uint8)  # Return black image on error

def main():
    """Main processing pipeline with configurable parameters."""
    try:
        # Load configuration
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print("Error: config.json file not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing config.json: {e}")
            sys.exit(1)

        # Validate and extract parameters
        required_keys = ['input_dir', 'output_dir', 'model_path']
        for key in required_keys:
            if key not in config:
                print(f"Missing required key in config: {key}")
                sys.exit(1)

        input_dir = Path(config['input_dir'])
        output_dir = Path(config['output_dir'])
        model_path = Path(config['model_path'])
        colormap_name = config.get('colormap_name', 'plasma')
        scale_factor = config.get('scale_factor', 4)

        try:
            scale_factor = int(scale_factor)
            if not 1 <= scale_factor <= 4:
                raise ValueError
        except (ValueError, TypeError):
            print("Invalid scale_factor - must be integer between 1-4")
            sys.exit(1)

        # Create output directories
        output_subdir = f"superres_{colormap_name}"
        for subdir in ["original", "superres", output_subdir]:
            (output_dir/subdir).mkdir(parents=True, exist_ok=True)

        # Initialize super-resolution model
        model = None
        if model_path.exists():
            try:
                model = cv2.dnn_superres.DnnSuperResImpl_create()
                model.readModel(str(model_path))
                model.setModel("edsr", scale_factor)
                print(f"Loaded model: {model_path}")
            except Exception as e:
                print(f"Model error: {str(e)}")
                model = None

        # Generate colormap LUT
        try:
            selected_cmap = plt.colormaps[colormap_name]
        except KeyError:
            print(f"Colormap '{colormap_name}' not found, using 'plasma'")
            selected_cmap = plt.colormaps['plasma']
        
        colormap_lut = (selected_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)[:, ::-1]
        colormap_lut = np.ascontiguousarray(colormap_lut)

        # Process files
        irg_files = list(input_dir.glob("*.irg"))
        if not irg_files:
            raise FileNotFoundError(f"No .irg files in {input_dir}")

        for file in tqdm(irg_files, desc="Processing"):
            try:
                data = file.read_bytes()
                coarse, _, _ = infiray_irg.load(data)
                
                if coarse is None or not isinstance(coarse, np.ndarray) or coarse.size == 0:
                    raise ValueError("Invalid .irg data")

                # Save original
                processed = apply_colormap(coarse, colormap_lut)
                cv2.imwrite(str(output_dir/f"original/{file.stem}_original.png"), processed)

                # Super-Resolution processing
                if model:
                    input_norm = cv2.normalize(coarse, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    input_rgb = cv2.cvtColor(input_norm, cv2.COLOR_GRAY2BGR)
                    result_rgb = model.upsample(input_rgb)
                    result_gray = cv2.cvtColor(result_rgb, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(str(output_dir/f"superres/{file.stem}_x{scale_factor}.png"), result_gray)
                    cv2.imwrite(str(output_dir/f"{output_subdir}/{file.stem}_{colormap_name}_x{scale_factor}.png"), 
                              apply_colormap(result_gray, colormap_lut))
                else:
                    h, w = coarse.shape
                    upscaled = cv2.resize(coarse, (w*scale_factor, h*scale_factor),
                                         interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(str(output_dir/f"superres/{file.stem}_x{scale_factor}.png"), upscaled)
                    cv2.imwrite(str(output_dir/f"{output_subdir}/{file.stem}_{colormap_name}_x{scale_factor}.png"),
                              apply_colormap(upscaled, colormap_lut))

            except Exception as e:
                print(f"\nError in {file.name}: {traceback.format_exc()}")
                continue

        input("\nProcessing complete! Press Enter...")

    except Exception as e:
        print(f"CRITICAL ERROR: {traceback.format_exc()}")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()