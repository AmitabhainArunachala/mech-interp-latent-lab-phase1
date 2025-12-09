#!/usr/bin/env python3
"""
Process image for passport photo: white background, proper dimensions
Uses flood-fill from corners to detect and replace background
"""

from PIL import Image, ImageDraw, ImageFilter
import sys
import os

def remove_background_floodfill(image_path, output_path, quality=85):
    """Remove background using flood-fill from corners"""
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    
    # Create a mask for background
    # Start flood-fill from corners
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Get corner colors
    corners = [
        img.getpixel((0, 0)),
        img.getpixel((width-1, 0)),
        img.getpixel((0, height-1)),
        img.getpixel((width-1, height-1))
    ]
    
    # Use average corner color as background color
    bg_color = tuple(sum(c[i] for c in corners) // len(corners) for i in range(3))
    
    # Threshold for similar colors
    threshold = 40
    
    # Create mask by comparing each pixel to background
    for y in range(height):
        for x in range(width):
            pixel = img.getpixel((x, y))
            # Calculate color distance
            dist = sum(abs(pixel[i] - bg_color[i]) for i in range(3))
            if dist < threshold:
                mask.putpixel((x, y), 255)  # Mark as background
    
    # Smooth the mask to avoid jagged edges
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Apply mask: white background where mask is white
    result = Image.new('RGB', (width, height), (255, 255, 255))
    
    for y in range(height):
        for x in range(width):
            mask_val = mask.getpixel((x, y))
            if mask_val < 128:  # Foreground (subject)
                result.putpixel((x, y), img.getpixel((x, y)))
            # else: keep white background
    
    # Save with compression
    result.save(output_path, 'JPEG', quality=quality, optimize=True)
    
    file_size = os.path.getsize(output_path) / 1024
    print(f"Processed image saved to: {output_path}")
    print(f"Size: {file_size:.1f} KB")
    print(f"Dimensions: {width}x{height} pixels")
    
    return file_size

if __name__ == "__main__":
    input_file = "/Users/dhyana/Downloads/IMG_7551.jpg"
    output_file = "/Users/dhyana/Downloads/IMG_7551_passport.jpg"
    
    # Try different quality levels to get under 1MB
    for quality in [90, 85, 80, 75]:
        size = remove_background_floodfill(input_file, output_file, quality=quality)
        if size < 1024:  # Under 1MB
            print(f"\n✓ Success! File is under 1MB at quality {quality}")
            break
    else:
        print(f"\n⚠ File is still {size:.1f} KB. You may need to resize the image.")
