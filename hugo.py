"""
hugo.py - HUGO Steganography Implementation
"""

from AES_LSB import UniversalSteganography


class HugoSteganography:
    """HUGO Steganography - Uses LSB as base implementation"""
    
    def __init__(self, payload=0.3):
        self.payload = payload
        self.lsb = UniversalSteganography(payload)
    
    def embed_file(self, cover_path, output_path, data, key):
        """
        Embed using HUGO algorithm
        Note: This is a simplified version using LSB.
        For production, implement the full HUGO algorithm with distortion-limited embedding.
        """
        return self.lsb.embed_file(cover_path, output_path, data, key)
    
    def extract_file(self, stego_path, key):
        """Extract data from HUGO stego image"""
        return self.lsb.extract_file(stego_path, key)