from transformers import BertTokenizer, BertForTokenClassification, pipeline

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model = BertForTokenClassification.from_pretrained('path/to/trained_model')
tokenizer = BertTokenizer.from_pretrained('path/to/trained_model')

# สร้าง pipeline สำหรับ NER
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer)

# ข้อความตัวอย่าง
text = "นายสมชาย บัณฑิตจากมหาวิทยาลัยกรุงเทพ"
ner_results = nlp_ner(text)

# แสดงผล
for entity in ner_results:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}")
