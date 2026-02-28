from thai_ocr.charset_thai_v1 import create_tokenizer

tokenizer = create_tokenizer()
print("num_classes =", len(tokenizer.id2ch))