# Disable oneDNN BEFORE any imports to avoid compatibility issues
import os
os.environ['PADDLE_ENABLE_ONEDNN'] = '0'
os.environ['PADDLE_CUDNN_DETERMINISTIC'] = '0'
os.environ['PADDLE_MKL_NUM_THREADS'] = '1'
os.environ['PADDLE_NUM_THREADS'] = '4'
os.environ['PADDLE_DISABLE_FAST_OPERATORS'] = '1'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from pathlib import Path
from paddleocr import PaddleOCR

# Create test_results folder if it doesn't exist
output_dir = "test_results"
os.makedirs(output_dir, exist_ok=True)

# Initialize OCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# Scan and process all images in test_images folder
test_images_dir = "test_images"
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

for image_file in sorted(os.listdir(test_images_dir)):
    if image_file.lower().endswith(image_extensions):
        image_path = os.path.join(test_images_dir, image_file)
        print(f"\nProcessing: {image_file}")
        
        try:
            # Run OCR inference
            result = ocr.predict(input=image_path)
            
            # Visualize the results and save the JSON results
            for res in result:
                res.print()
                res.save_to_img(output_dir)
                res.save_to_json(output_dir)
            
            print(f"[SUCCESS] Results saved for: {image_file}")
        except Exception as e:
            print(f"[ERROR] Error processing {image_file}: {str(e)}")
            continue

print("\nProcessing complete!")
