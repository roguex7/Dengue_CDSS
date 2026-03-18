#save as verify_ocr.py and run: python verify_ocr.py
try:
    import pytesseract
    print("✅ pytesseract installed:", pytesseract.get_tesseract_version())
except Exception as e:
    print("❌ pytesseract error:", e)

try:
    import cv2
    print("✅ OpenCV installed:", cv2.__version__)
except ImportError:
    print("❌ OpenCV not installed: pip install opencv-python")

try:
    from PIL import Image
    print("✅ Pillow installed")
except ImportError:
    print("❌ Pillow not installed: pip install pillow")

try:
    from pdf2image import convert_from_bytes
    print("✅ pdf2image installed")
except ImportError:
    print("❌ pdf2image not installed: pip install pdf2image")

print("\n📋 Summary: All checks above must show ✅ for full OCR functionality.")