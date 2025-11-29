import json
import re

# Load the notebook
with open('labs/CIA3/2447137_CAI3.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell with sobel_edge_detection function
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def sobel_edge_detection' in source:
            # Replace the function
            new_source = re.sub(
                r'(def sobel_edge_detection\(image\):\s*# Sobel kernels\s*sobel_x = np\.array\(\[\[-1, 0, 1\], \[-2, 0, 2\], \[-1, 0, 1\]\]\)\s*sobel_y = np\.array\(\[\[-1, -2, -1\], \[0, 0, 0\], \[1, 2, 1\]\]\))',
                r'\1\n    sobel_xy = sobel_x + sobel_y',
                source
            )
            # Add edge_xy computation
            new_source = re.sub(
                r'(edge_y = ndimage\.convolve\(image\.astype\(float\), sobel_y\)\s*# Magnitude)',
                r'\1\n    edge_xy = ndimage.convolve(image.astype(float), sobel_xy)\n    \n    # Magnitude',
                new_source
            )
            # Update return
            new_source = re.sub(
                r'return edge_x, edge_y, edge_magnitude',
                r'return edge_x, edge_y, edge_xy, edge_magnitude',
                new_source
            )
            cell['source'] = new_source.split('\n')
            break

# Save the notebook
with open('labs/CIA3/2447137_CAI3.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
