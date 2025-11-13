"""
wow.py - WOW Steganography Implementation
"""

from AES_LSB import UniversalSteganography


class WowSteganography:
    """WOW Steganography - Uses LSB as base implementation"""
    
    def __init__(self, payload=0.3):
        self.payload = payload
        self.lsb = UniversalSteganography(payload)
    
    def embed_file(self, cover_path, output_path, data, key):
        """
        Embed using WOW algorithm
        Note: This is a simplified version using LSB.
        For production, implement the full WOW algorithm with wavelet-obtained weights.
        """
        return self.lsb.embed_file(cover_path, output_path, data, key)
    
    def extract_file(self, stego_path, key):
        """Extract data from WOW stego image"""
        return self.lsb.extract_file(stego_path, key)