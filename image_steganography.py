"""
image_steganography.py - Hide Images/Data Inside Images
Uses 2-bit LSB steganography with metadata
"""

import numpy as np
from PIL import Image
import io
import math


class ImageSteganography:
    """Advanced LSB steganography for hiding images inside images"""
    
    def __init__(self):
        pass
    
    def embed_image(self, cover_path, secret_path, output_path):
        """
        Embed a secret image inside a cover image
        Returns: (success, error_message)
        """
        try:
            # Load cover image
            cover_img = Image.open(cover_path)
            cover_array = np.array(cover_img.convert('RGB'))
            
            # Load secret image
            secret_img = Image.open(secret_path)
            
            # Convert secret image to bytes (PNG format preserves quality)
            secret_bytes_io = io.BytesIO()
            secret_img.save(secret_bytes_io, format='PNG')
            secret_data = secret_bytes_io.getvalue()
            
            # Get filename
            import os
            filename = os.path.basename(secret_path)
            if len(filename) > 12:
                filename = filename[:12]  # Truncate if too long
            
            # Calculate capacity
            max_bytes = self._calculate_capacity(cover_array)
            
            if len(secret_data) > max_bytes - 17:  # 17 bytes for metadata
                return False, f'Secret image too large! Max: {(max_bytes-17)//1024}KB, Yours: {len(secret_data)//1024}KB'
            
            # Embed data
            stego_array = self._write_data(cover_array, secret_data, filename, data_type='image')
            
            # Save stego image
            stego_img = Image.fromarray(stego_array.astype(np.uint8))
            stego_img.save(output_path, 'PNG')
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def embed_data_bytes(self, cover_path, data_bytes, output_path, filename="data.bin"):
        """
        Embed arbitrary bytes (for text or other data) inside a cover image
        Returns: (success, error_message)
        """
        try:
            # Load cover image
            cover_img = Image.open(cover_path)
            cover_array = np.array(cover_img.convert('RGB'))
            
            # Ensure filename is valid
            if len(filename) > 12:
                filename = filename[:12]
            
            # Calculate capacity
            max_bytes = self._calculate_capacity(cover_array)
            
            if len(data_bytes) > max_bytes - 17:
                return False, f'Data too large! Max: {max_bytes-17} bytes, Yours: {len(data_bytes)} bytes'
            
            # Embed data (mark as 'text' type for generic data)
            stego_array = self._write_data(cover_array, data_bytes, filename, data_type='text')
            
            # Save stego image
            stego_img = Image.fromarray(stego_array.astype(np.uint8))
            stego_img.save(output_path, 'PNG')
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def extract_image(self, stego_path):
        """
        Extract hidden data from stego image
        Returns: (data_bytes, filename, data_type) or (None, None, None) on failure
        """
        try:
            # Load stego image
            stego_img = Image.open(stego_path)
            stego_array = np.array(stego_img.convert('RGB'))
            
            # Extract data
            data, filename, data_type = self._extract_data(stego_array)
            
            if data is None:
                return None, None, None
            
            return bytes(data), filename, data_type
            
        except Exception as e:
            print(f"Extraction error: {e}")
            return None, None, None
    
    def _calculate_capacity(self, img_array):
        """Calculate maximum bytes that can be hidden"""
        height, width, channels = img_array.shape
        # Using 2 bits per channel, 3 channels per pixel
        # Each byte needs 4 channel values (8 bits / 2 bits per channel)
        return math.floor(height * width * channels * 2 / 8)
    
    def _write_data(self, img, data, filename, data_type='image'):
        """
        Embeds data into an image using 2-bit LSB steganography
        data_type: 'image' or 'text'
        """
        byte_array = bytearray()
        
        # First byte indicates type: 0=image, 1=text
        type_byte = 1 if data_type == 'text' else 0
        byte_array.append(type_byte)
        
        # Next 12 bytes for filename (padded with '0')
        byte_array.extend(bytes(filename.rjust(12, '0'), 'utf-8'))
        
        # Next 4 bytes for data length (little endian)
        byte_array.extend(len(data).to_bytes(4, 'little'))
        
        # Then the actual data
        byte_array.extend(data)
        
        height, width, channels = img.shape
        data_size = len(byte_array)
        byte_num = 0
        nib_num = 0  # Nibble counter (0-3, since we use 2 bits at a time)
        
        # Embed data
        for i in range(height):
            for j in range(width):
                for c in range(channels):
                    # Clear lowest 2 bits and insert 2 bits from data
                    img[i, j, c] = (img[i, j, c] & 0xFC) | ((byte_array[byte_num] >> (nib_num * 2)) & 0x03)
                    
                    nib_num += 1
                    if nib_num == 4:  # We've embedded all 4 pairs of bits from this byte
                        nib_num = 0
                        byte_num += 1
                        
                        if byte_num >= data_size:
                            return img
        
        return img
    
    def _extract_data(self, img):
        """
        Extracts hidden data from a stego image
        Returns: (data, filename, data_type)
        """
        type_byte_array = bytearray()
        filename_byte_array = bytearray()
        filesize_byte_array = bytearray()
        byte_array = bytearray()
        
        height, width, channels = img.shape
        data_size = 17  # 1 (type) + 12 (filename) + 4 (size)
        byte_count = 0
        nib = 0  # Nibble counter
        byte_dat = 0
        
        # Extract data
        for i in range(height):
            for j in range(width):
                for c in range(channels):
                    # Extract 2 bits from LSB
                    byte_dat = byte_dat | ((img[i, j, c] & 0x03) << (nib * 2))
                    
                    nib += 1
                    if nib == 4:  # We've collected all 4 pairs of bits for this byte
                        nib = 0
                        
                        # Store in appropriate array
                        if byte_count == 0:
                            type_byte_array.append(byte_dat)
                        elif byte_count < 13:
                            filename_byte_array.append(byte_dat)
                        elif byte_count < 17:
                            filesize_byte_array.append(byte_dat)
                            if byte_count == 16:
                                # Now we know the total data size
                                data_size = int.from_bytes(filesize_byte_array, 'little') + 17
                        else:
                            byte_array.append(byte_dat)
                        
                        byte_dat = 0
                        byte_count += 1
                        
                        if byte_count >= data_size:
                            # Extraction complete
                            data_type = 'text' if type_byte_array[0] == 1 else 'image'
                            filename = filename_byte_array.decode('utf-8').lstrip('0')
                            return byte_array, filename, data_type
        
        return None, None, None