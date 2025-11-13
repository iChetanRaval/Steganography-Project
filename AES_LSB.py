"""
AES_LSB.py - LSB Steganography with AES Encryption
"""

import numpy as np
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


class UniversalSteganography:
    """LSB Steganography with AES encryption"""
    
    def __init__(self, payload=0.3):
        self.payload = payload
    
    def embed_file(self, cover_path, output_path, data, key):
        """Embed data into image using LSB"""
        try:
            # Load cover image
            img = Image.open(cover_path)
            img_array = np.array(img.convert('RGB'))
            
            # Encrypt data
            encrypted_data = self._encrypt(data, key)
            
            # Convert to binary
            data_length = len(encrypted_data)
            binary_data = self._to_binary(data_length, 32) + self._bytes_to_binary(encrypted_data)
            
            # Check capacity
            max_bytes = img_array.size // 8
            if len(binary_data) > max_bytes * 8:
                print(f"Error: Data too large. Max: {max_bytes} bytes, Need: {len(binary_data)//8} bytes")
                return False
            
            # Embed data using LSB
            flat_img = img_array.flatten()
            for i, bit in enumerate(binary_data):
                flat_img[i] = (flat_img[i] & 0xFE) | int(bit)
            
            # Reshape and save
            stego_array = flat_img.reshape(img_array.shape)
            stego_img = Image.fromarray(stego_array.astype(np.uint8))
            stego_img.save(output_path)
            
            print(f"✅ Successfully embedded {len(data)} characters")
            return True
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return False
    
    def extract_file(self, stego_path, key):
        """Extract data from stego image"""
        try:
            # Load stego image
            img = Image.open(stego_path)
            img_array = np.array(img.convert('RGB'))
            flat_img = img_array.flatten()
            
            # Extract data length (first 32 bits)
            length_bits = ''.join([str(pixel & 1) for pixel in flat_img[:32]])
            data_length = int(length_bits, 2)
            
            # Validate data length
            if data_length <= 0 or data_length > len(flat_img) // 8:
                print(f"❌ Invalid data length: {data_length}")
                return None
            
            # Extract encrypted data
            total_bits = 32 + (data_length * 8)
            if total_bits > len(flat_img):
                print(f"❌ Data length exceeds image capacity")
                return None
            
            data_bits = ''.join([str(pixel & 1) for pixel in flat_img[32:total_bits]])
            
            # Convert binary to bytes
            encrypted_data = self._binary_to_bytes(data_bits)
            
            # Decrypt data
            decrypted_data = self._decrypt(encrypted_data, key)
            
            print(f"✅ Successfully extracted {len(decrypted_data)} characters")
            return decrypted_data
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            return None
    
    def _encrypt(self, data, key):
        """Encrypt data using AES"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        cipher = AES.new(key, AES.MODE_CBC)
        ct_bytes = cipher.encrypt(pad(data, AES.block_size))
        return cipher.iv + ct_bytes
    
    def _decrypt(self, encrypted_data, key):
        """Decrypt data using AES"""
        iv = encrypted_data[:16]
        ct = encrypted_data[16:]
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted = unpad(cipher.decrypt(ct), AES.block_size)
        return decrypted.decode('utf-8')
    
    def _to_binary(self, value, bits):
        """Convert integer to binary string"""
        return format(value, f'0{bits}b')
    
    def _bytes_to_binary(self, data):
        """Convert bytes to binary string"""
        return ''.join(format(byte, '08b') for byte in data)
    
    def _binary_to_bytes(self, binary_str):
        """Convert binary string to bytes"""
        byte_array = bytearray()
        for i in range(0, len(binary_str), 8):
            byte = binary_str[i:i+8]
            if len(byte) == 8:
                byte_array.append(int(byte, 2))
        return bytes(byte_array)