from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset

# โหลดโมเดลและ tokenizer
model = BertForTokenClassification.from_pretrained("phate/bert-base-thai")
tokenizer = BertTokenizer.from_pretrained("phate/bert-base-thai")

# โหลดและเตรียมข้อมูล
dataset = load_dataset('json', data_files={'train': 'data/train.json', 'test': 'data/test.json'})

# กำหนดการฝึก
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer
)

# ฝึกโมเดล
trainer.train()
