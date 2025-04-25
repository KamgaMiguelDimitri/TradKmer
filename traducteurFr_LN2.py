import pandas as pd
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
from datasets import Dataset,DatasetDict



# Charger le dataset
data = pd.read_csv('datasetFr_Ln.csv')  # Modifiez le chemin selon votre fichier


# dataFrame des donnees d'entrainement
train_df = data.sample(frac=0.8)

# dataFrame des donnees de test
#test_df = data.drop(train_df.index)
test_df = data.sample(frac=0.2)

#transformation des donnees de test et d'entrainement en dataSet
train_set = Dataset.from_pandas(train_df)
test_set = Dataset.from_pandas(test_df)

dataset = DatasetDict({'train':train_set,
                       'test':test_set})

# Charger le modèle et le tokenizer
model_name = 'Helsinki-NLP/opus-mt-fr-ln'  # Modèle pré-entraîné
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def preprocess_funtion(examples):
    inputs = [ex for ex in examples['Francais']]
    targets = [ex for ex in examples['Lingala']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
    labels = tokenizer(targets,max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_dataset = dataset.map(preprocess_funtion, batched=True)

# Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
)

# Entraînement du modèle
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test']
)

trainer.train()

#enregistrer le modele pour l'utiliser apres

model.save_pretrained("./model_fr_lingala")
tokenizer.save_pretrained("./model_fr_lingala")


def translate(text):
    inputs = tokenizer.encode( text,
                              return_tensors="pt", max_length=128, truncation=True)
    
    outputs = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

#Exemple d'utilisation

print(translate("Bonjour je m'appelle miguel et toi"))