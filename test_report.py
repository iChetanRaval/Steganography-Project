"""
test_report.py - Test script for forensic report generation
Run this to verify the integration works correctly
"""

import os
from forensic_report import ForensicReportGenerator

def test_report_generation():
    """Test the forensic report generation with sample images"""
    
    print("\n" + "="*70)
    print("🧪 FORENSIC REPORT GENERATION TEST")
    print("="*70 + "\n")
    
    # Initialize generator
    print("1️⃣ Initializing ForensicReportGenerator...")
    generator = ForensicReportGenerator()
    print("   ✅ Generator initialized\n")
    
    # Check for test images
    print("2️⃣ Checking for test images...")
    
    # You can modify these paths to your actual test images
    test_cases = [
        {
            "name": "LSB Test",
            "cover": "test_images/cover.jpg",
            "stego": "test_images/stego_lsb.png",
            "algorithm": "LSB",
            "confidence": 95.5
        },
        {
            "name": "HUGO Test",
            "cover": "test_images/cover.jpg",
            "stego": "test_images/stego_hugo.jpg",
            "algorithm": "HUGO",
            "confidence": 92.3
        },
        {
            "name": "Clean Image Test",
            "cover": "test_images/clean.jpg",
            "stego": "test_images/clean.jpg",
            "algorithm": None,
            "confidence": 100.0
        }
    ]
    
    # Find available test cases
    available_tests = []
    for test in test_cases:
        if os.path.exists(test["cover"]) and os.path.exists(test["stego"]):
            available_tests.append(test)
            print(f"   ✅ Found: {test['name']}")
        else:
            print(f"   ⏭️  Skipped: {test['name']} (images not found)")
    
    if not available_tests:
        print("\n⚠️  No test images found!")
        print("\n💡 To test with your own images:")
        print("   1. Create a 'test_images' folder")
        print("   2. Add cover.jpg and stego images")
        print("   3. Or modify the test_cases paths above")
        print("\n🔄 Attempting to test with same image (self-comparison)...")
        
        # Try to find any image in the project
        possible_paths = [
            "static/generated/",
            "uploads/",
            "test_images/",
            "./"
        ]
        
        test_image = None
        for path in possible_paths:
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        test_image = os.path.join(path, file)
                        break
            if test_image:
                break
        
        if test_image:
            print(f"   ✅ Found test image: {test_image}")
            available_tests = [{
                "name": "Self-Comparison Test",
                "cover": test_image,
                "stego": test_image,
                "algorithm": None,
                "confidence": 100.0
            }]
        else:
            print("\n❌ No images found for testing")
            print("\nPlease provide at least one test image and update this script.")
            return
    
    print(f"\n3️⃣ Running {len(available_tests)} test(s)...\n")
    
    # Run tests
    results = []
    for i, test in enumerate(available_tests, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}/{len(available_tests)}: {test['name']}")
        print(f"{'─'*70}")
        print(f"📂 Cover:  {test['cover']}")
        print(f"📂 Stego:  {test['stego']}")
        print(f"🔐 Algorithm: {test['algorithm'] or 'None (Clean)'}")
        print(f"📊 Confidence: {test['confidence']}%")
        print()
        
        # Generate output filename
        output_pdf = f"test_report_{i}_{test['name'].replace(' ', '_')}.pdf"
        
        print(f"⚙️  Generating report: {output_pdf}")
        
        # Generate report
        success, error = generator.generate_report(
            cover_path=test["cover"],
            stego_path=test["stego"],
            output_pdf_path=output_pdf,
            detected_algorithm=test["algorithm"],
            confidence=test["confidence"]
        )
        
        if success:
            print(f"✅ SUCCESS! Report saved to: {output_pdf}")
            file_size = os.path.getsize(output_pdf) / 1024
            print(f"📄 File size: {file_size:.1f} KB")
            results.append({
                "test": test["name"],
                "status": "✅ SUCCESS",
                "output": output_pdf
            })
        else:
            print(f"❌ FAILED: {error}")
            results.append({
                "test": test["name"],
                "status": f"❌ FAILED: {error}",
                "output": None
            })
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70 + "\n")
    
    for result in results:
        print(f"{result['status']} - {result['test']}")
        if result['output']:
            print(f"   📄 {result['output']}")
        print()
    
    success_count = sum(1 for r in results if "SUCCESS" in r["status"])
    print(f"Total: {success_count}/{len(results)} tests passed")
    
    if success_count == len(results):
        print("\n🎉 All tests passed! The integration is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    # Cleanup
    print("\n4️⃣ Cleaning up temporary files...")
    generator.cleanup()
    print("   ✅ Cleanup complete")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        test_report_generation()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()