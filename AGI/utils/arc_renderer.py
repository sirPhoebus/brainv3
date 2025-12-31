import numpy as np
from PIL import Image
import os

# ARC color palette [0-9]
ARC_COLORS = [
    (0, 0, 0),       # 0: black
    (0, 116, 217),   # 1: blue
    (255, 65, 54),   # 2: red
    (46, 204, 64),   # 3: green
    (255, 220, 0),   # 4: yellow
    (170, 170, 170), # 5: gray
    (240, 18, 190),  # 6: magenta
    (255, 133, 27),  # 7: orange
    (127, 219, 255), # 8: azure
    (135, 12, 37)    # 9: maroon
]

def render_grid(grid, cell_size=20):
    """Render an ARC grid to a PIL Image."""
    grid = np.array(grid)
    height, width = grid.shape
    img_data = np.zeros((height * cell_size, width * cell_size, 3), dtype=np.uint8)
    
    for r in range(height):
        for c in range(width):
            color = ARC_COLORS[grid[r, c]]
            img_data[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] = color
            
    return Image.fromarray(img_data)

def render_task_pairs(task, output_dir, task_id="arc_sample"):
    """
    Render input/output pairs for an ARC task.
    Concatenates input and output horizontally for the cortex to see the relationship.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Process training pairs
    # For now, let's just create one composite image of the example provided
    input_img = render_grid(task['input'])
    output_img = render_grid(task['output'])
    
    # Create horizontal composite
    combined = Image.new('RGB', (input_img.width + output_img.width + 10, input_img.height))
    combined.paste(input_img, (0, 0))
    combined.paste(output_img, (input_img.width + 10, 0))
    
    path = os.path.join(output_dir, f"{task_id}_composite.png")
    combined.save(path)
    print(f"Saved ARC composite to {path}")
    return path

if __name__ == "__main__":
    # Example task from user
    task_example = {
        "input": [[0]*7 + [3, 3, 3] + [0]*6 for _ in range(16)],
        "output": [[0]*16 for _ in range(16)] # placeholder for user's complex output
    }
    # Update with user's specific data
    task_example['input'][7] = [0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0]
    
    # The user's output grid is complex, I'll just use the logic described
    # For the sake of the renderer test, let's just render the example provided
    user_task = {
        "input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
        "output": [[0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0], [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0], [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0], [0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0], [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 0, 0], [0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0], [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0], [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0], [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0], [3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]]
    }
    render_task_pairs(user_task, "AGI/examples/arc_tasks", "task_user")
