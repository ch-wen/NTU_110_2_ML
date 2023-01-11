# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 18:17:22 2022

@author: MD304
"""

import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset 
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast, get_linear_schedule_with_warmup
import math
from tqdm.auto import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Fix random seed for reproducibility
def same_seeds(seed):
	  torch.manual_seed(seed)
	  if torch.cuda.is_available():
		    torch.cuda.manual_seed(seed)
		    torch.cuda.manual_seed_all(seed)
	  np.random.seed(seed)
	  random.seed(seed)
	  torch.backends.cudnn.benchmark = False
	  torch.backends.cudnn.deterministic = True
same_seeds(0)

# Change "fp16_training" to True to support automatic mixed precision training (fp16)	
fp16_training = False

if fp16_training:
    from accelerate import Accelerator
    accelerator = Accelerator(fp16=True)
    device = accelerator.device

# Documentation for the toolkit:  https://huggingface.co/docs/accelerate/
model = BertForQuestionAnswering.from_pretrained("hfl/chinese-roberta-wwm-ext-large").to(device)
tokenizer = BertTokenizerFast.from_pretrained("hfl/chinese-roberta-wwm-ext-large")

# You can safely ignore the warning message (it pops up because new prediction heads for QA are initialized randomly)

"""## Read Data
- Training set: 26935 QA pairs
- Dev set: 3523  QA pairs
- Test set: 3492  QA pairs
- {train/dev/test}_questions:	
  - List of dicts with the following keys:
   - id (int)
   - paragraph_id (int)
   - question_text (string)
   - answer_text (string)
   - answer_start (int)
   - answer_end (int)
- {train/dev/test}_paragraphs: 
  - List of strings
  - paragraph_ids in questions correspond to indexs in paragraphs
  - A paragraph may be used by several questions 
"""

def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]

train_questions, train_paragraphs = read_data("hw7_train.json")
dev_questions, dev_paragraphs = read_data("hw7_dev.json")
test_questions, test_paragraphs = read_data("hw7_test.json")

"""## Tokenize Data"""

# Tokenize questions and paragraphs separately
# 「add_special_tokens」 is set to False since special tokens will be added when tokenized questions and paragraphs are combined in datset __getitem__ 

train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=False)
test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False) 
'''
max_len = 0
for question in [*train_questions_tokenized.input_ids, *dev_questions_tokenized.input_ids, *test_questions_tokenized.input_ids]:
    max_len = max(max_len, len(question))
print(f"Max question length: {max_len}")
'''
train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)

# You can safely ignore the warning message as tokenized sequences will be futher processed in datset __getitem__ before passing to model

"""## Dataset and Dataloader"""

class QA_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = 200
        self.max_paragraph_len = 300
        
        ##### TODO: Change value of doc_stride #####
        self.doc_stride = 260

        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]

        ##### TODO: Preprocessing #####
        # Hint: How to prevent model from learning something it should not learn

        if self.split == "train":
            # Convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph  
            answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
            answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])

            # A single window is obtained by slicing the portion of paragraph containing the answer
            ans_len = (answer_end_token - answer_start_token) + 1
            paragraph_start = max(0, min(answer_start_token - random.randint(0, self.max_paragraph_len - ans_len), len(tokenized_paragraph) - self.max_paragraph_len))
            paragraph_end = paragraph_start + self.max_paragraph_len
            
            # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102] 
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start : paragraph_end] + [102]		
            
            # Convert answer's start/end positions in tokenized_paragraph to start/end positions in the window  
            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start
            
            # Pad sequence and obtain inputs to model 
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token

        # Validation/Testing
        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
            start_id = []
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i : i + self.max_paragraph_len] + [102]
                
                # Pad sequence and obtain inputs to model
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
                start_id.append(i)
            
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list), start_id, question["paragraph_id"]

    def padding(self, input_ids_question, input_ids_paragraph):
        # Pad zeros if sequence length is shorter than max_seq_len
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        # Indices of input sequence tokens in the vocabulary
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        
        return input_ids, token_type_ids, attention_mask

train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)

train_batch_size = 2

# Note: Do NOT change batch size of dev_loader / test_loader !
# Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

"""## Function for Evaluation"""

def evaluate(data, output, content, token_content):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing 
    # Hint: Open your prediction file to see what is wrong 
   # p = data[-1]
    start_id = data[-2]
    
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    question_len = 0
    
    for i in data[0][0][0]:
        question_len+=1
        if i.item() == 102:
            break
    
    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        #mask = data[1][0][k].bool() &  data[2][0][k].bool() # token type & attention mask
        #mask.to("cuda")
        #masked_output_start = torch.masked_select(output.start_logits[k], mask)[:-1] # -1 is [SEP]
        #masked_output_start.to("cuda")
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        #start_prob, start_index = torch.max(masked_output_start, dim=0)
        
        #masked_output_end = torch.masked_select(output.end_logits[k], mask)[start_index:-1] # -1 is [SEP]
        #masked_output_end.to("cuda")
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)
        #end_prob, end_index = torch.max(masked_output_end, dim=0)
        
        #end_index += start_index
        
        prob = start_prob + end_prob
        #masked_data = torch.masked_select(data[0][0][k], mask)[:-1] # -1 is [SEP]
        
        if end_index < start_index:
            prob = 0
            continue
        
        # Probability of answer is calculated as sum of start_prob and end_prob
        #prob = start_prob + end_prob
        
        # Replace answer if calculated probability is larger than previous windows
        if (prob > max_prob) and (end_index - start_index <= 30) and (end_index > start_index):
            max_prob = prob
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            #answer = tokenizer.decode(masked_data[start_index : end_index + 1])
            answer = tokenizer.decode(data[0][0][k][start_index: end_index + 1])
            answer = answer.replace(" ", "")
            answer = answer.replace("[PAD]", "")
            answer = answer.replace("[SEP]", "")
            data_l = data[0][0][k][start_index: end_index + 1].tolist()
            while "[UNK]" in answer:
                #print(answer)
                unk_token_idx = data_l.index(100)
                data_l[unk_token_idx] = 0
                #print(unk_token_idx + start_index.item() - question_len + start_id[k].item())
                idx = token_content.token_to_chars(abs(unk_token_idx + start_index.item() - question_len + start_id[k].item()))
                answer = answer.replace("[UNK]", content[idx[0]:idx[1]])
                #print(answer)
    
    # Remove spaces in answer (e.g. "大 金" --> "大金")
    return answer

"""## Training"""

num_epoch = 15
validation = True
logging_step = 200
accu_step = 64
num_warmup_steps = 1000
num_train_steps = len(train_loader.dataset) * num_epoch / train_batch_size
learning_rate = 1e-5
optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False, betas=(0.9, 0.98))
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)


if fp16_training:
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader) 

try:
    print(f"Loading best model")
    model.load_state_dict(torch.load("model.ckpt", map_location=device))
except:
    print("cannot load model")
model.train()

print("Start Training ...")

for epoch in range(num_epoch):
    step = 1
    train_loss = train_acc = 0
    
    for data in tqdm(train_loader):	
        # Load all data into GPU
        data = [i.to(device) for i in data]
        
        # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
        # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)  
        output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])

        # Choose the most probable start position / end position
        start_index = torch.argmax(output.start_logits, dim=1)
        end_index = torch.argmax(output.end_logits, dim=1)
        
        # Prediction is correct only if both start_index and end_index are correct
        train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
        train_loss += output.loss
        
        if fp16_training:
            accelerator.backward(output.loss)
        else:
            output.loss = output.loss / accu_step
            output.loss.backward()
        
        if step % accu_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        step += 1

        ##### TODO: Apply linear learning rate decay #####
        
        # Print training loss and accuracy over past logging step
        if step % logging_step == 0:
            print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
            train_loss = train_acc = 0

    if validation:
        print("Evaluating Dev Set ...")
        model.eval()
        with torch.no_grad():
            dev_acc = 0
            for i, data in enumerate(tqdm(dev_loader)):
                output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
                # prediction is correct only if answer text exactly matches
                dev_acc += evaluate(data, output, dev_paragraphs[data[-1]], dev_paragraphs_tokenized[data[-1]]) == dev_questions[i]["answer_text"]
            print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
            torch.save(model.state_dict(), f"./valid.{epoch + 1}.ckpt")
            torch.save(model.state_dict(), f"./model.ckpt")
        model.train()

# Save a model and its configuration file to the directory 「saved_model」 
# i.e. there are two files under the direcory 「saved_model」: 「pytorch_model.bin」 and 「config.json」
# Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("saved_model")」
print("Saving Model ...")
model_save_dir = "saved_model" 
model.save_pretrained(model_save_dir)

"""## Testing"""

print("Evaluating Test Set ...")

result = []

model.eval()
with torch.no_grad():
    for data in tqdm(test_loader):
        output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
        result.append(evaluate(data, output, test_paragraphs[data[-1].item()], test_paragraphs_tokenized[data[-1].item()]))

result_file = "result.csv"
with open(result_file, 'w', encoding="utf-8") as f:	
	  f.write("ID,Answer\n")
	  for i, test_question in enumerate(test_questions):
        # Replace commas in answers with empty strings (since csv is separated by comma)
        # Answers in kaggle are processed in the same way
		    f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

print(f"Completed! Result is in {result_file}")