import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import infiray_irg
import numpy as np
from tqdm import tqdm

def main():
    input_file = "example.irg"
    output_file = "all_colormaps.pdf"
    
    # Reading IRG-file data
    with open(input_file, 'rb') as f:
        irg_data = infiray_irg.load(f.read())
    
    thermal_data = irg_data[1]
    
    # Data normalization
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    normalized_data = normalize(thermal_data)
    
    #  PDF generation
    with PdfPages(output_file) as pdf:
        for cmap in tqdm(plt.colormaps(), desc="PDF generation", unit="colormap"):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            img = ax.imshow(normalized_data, cmap=cmap)
            ax.set_title(f"Colormap: {cmap}", fontsize=10)
            ax.axis('off')
            fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close()

if __name__ == "__main__":
    main()