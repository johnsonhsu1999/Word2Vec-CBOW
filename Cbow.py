
import torch
import nltk
from torch.utils.data import Dataset,DataLoader

BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

#1. vocab and corpus(dict index), include bos/eos token to index
def preprocess(corpus): #input is pure string article
    context = [sent for sent in nltk.sent_tokenize(corpus)]
    vocab = dict()
    vocab["<unk>"]=0
    vocab[BOS_TOKEN]=1
    vocab[EOS_TOKEN]=2
    token_to_word = ["<unk>",BOS_TOKEN,EOS_TOKEN]
    index = 3

    for sent in context:
        for word in nltk.word_tokenize(sent):
            if word not in vocab:
                token_to_word.append(word)
                vocab[word]=index
                index+=1
    tokens = [[vocab[word] for word in nltk.word_tokenize(sent)] for sent in context]
    
    return tokens, vocab, token_to_word


#2. make cbow dataset
class CbowDataset(Dataset):   #->[(上下文,中間詞),(上下文,中間詞),...,(上下文,中間詞)]
    #PAIR(self.data)= (上下文, 中間詞)
    #assume vocab is exist and corpus is number list [[1,2,4],[3,6,4,6,2],...,[]]
    #[_,_,target,_,_] if window size=2  --> from first(_,_,Target,_,_,...) to last(......,_,_,Target,_,_)
    def __init__(self, corpus, vocab, window_size):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in corpus:
            sentence = [self.bos] + sentence + [self.eos]
            if len(sentence)<window_size*2+1:  
                continue
            else:
                #from first(_,_,Target,_,_,...) to last(......,_,_,Target,_,_)
                for i in range(window_size,len(sentence)-window_size):
                    context = torch.tensor(sentence[i-window_size:i] + sentence[i+1:i+window_size+1])
                    target = sentence[i]
                    self.data.append((context,target))
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self,i):
        return self.data[i]

#3. Cbow model
from torch.nn import Module
class CbowModel(Module):
    def __init__(self,vocab,embedding_dim):
        super(CbowModel,self).__init__()
        self.embeddings = torch.nn.Embedding(len(vocab),embedding_dim=embedding_dim)
        self.outputs = torch.nn.Linear(embedding_dim, len(vocab))
    def forward(self,x):
        #print("input shape :",x.shape) #(8, 4)
        x = self.embeddings(x)  #dim=(l, V)->dim(d, V)
        #print("after embedding : ",x.shape) ([8, 4, 32])
        x = x.mean(axis=1)  #dim(d, V)->dim(1, V)
        #print("x.mean(axis=1) shape : ",x.shape) ([8, 32])!!!!這裡取上下文size的內容取平均當作壓平，才能use linear
        x = self.outputs(x) # ([8, 175]) batch_size=8, each output有175大小的分佈
        return x


#4. training Data
text = "Hello, my name is John. I'm a friendly and outgoing person who enjoys meeting new people. I have a passion for exploring different cultures and learning about their traditions. In my free time, I love playing sports, especially basketball and soccer. Let me tell you about Ivy, my amazing girlfriend. She is not only beautiful but also incredibly kind-hearted. Her warm smile can light up a room, and her caring nature makes everyone around her feel special. I feel incredibly lucky to have her in my life. We have shared many wonderful moments together. From our romantic walks on the beach to fun-filled adventures, every experience with Ivy is unforgettable. She brings so much joy and happiness into my life, and I'm grateful for every moment we spend together. Ivy's love and support have been a constant source of strength for me. She believes in my dreams and encourages me to pursue my passions. I can always count on her to be there for me, through both the ups and downs of life. Expressing my gratitude towards Ivy is of utmost importance to me. She has been my rock, my confidante, and my best friend. I want her to know how much she means to me and how grateful I am for her presence in my life. Ivy, thank you for being such an incredible girlfriend. You make my world brighter and my heart fuller. I cherish every moment we spend together and look forward to creating many more beautiful memories with you. In conclusion, the love I share with Ivy is something truly special. Our bond grows stronger with each passing day, and I'm excited about the future we will build together. I am grateful for her love, and I will continue to cherish and appreciate her every day."
tokens,vocab,token_to_word = preprocess(text)
cbow_dataset = CbowDataset(corpus=tokens,vocab=vocab,window_size=2)
cbow_dataLoader = DataLoader(dataset=cbow_dataset,batch_size=8,shuffle=False)
model = CbowModel(vocab=vocab,embedding_dim=32)

#5. training Model
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tqdm.auto import tqdm
device = torch.device("cpu")
criteria = CrossEntropyLoss()
optimizer = SGD(model.parameters(),lr=0.02)
epoch_num = 100
total_loss = 0
for epoch in range(epoch_num):
    losses = 0
    for batch in tqdm(cbow_dataLoader,desc="training "):
        context, target =batch
        #print("input shape : ",context.shape) #(8, 4)
        #print("target shape : ",target.shape) #(8)
        predict = model(context)
        #print("target shape : ",target.shape) #shape=(7)
        #print("predict shape : ",predict.shape) #shape=(7, 175)==>模型吐出的是一個prob distribution
        #predicted_labels = torch.argmax(predict, dim=1)  #important 看預測出哪個中間詞
        #ans = [token_to_word[w] for w in predicted_labels.tolist()] #每batch的預測詞
        loss = criteria(predict, target) #model輸出的是各詞的分佈, predict_shape=(1,175),target_shape=(1,)
        #print("loss item : ",loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses += loss.item()
    print(f"epoch : {epoch+1}, loss : {losses}")

#6. save "word vector"
word_vector = model.embeddings.weight
print(word_vector)
print(word_vector.shape) #(175, 32)


#7. testing
inputs = ['Let','me','you','about'] 
a = torch.tensor([vocab[word] if word in vocab else 0 for word in inputs]) #important #tensor([ 44,  22,  90, 168])
a = a.view(1, -1) #(1,4) 1 batch, 上下文共4    
predict = model(a)  #(1,4,175) 1 data, 上下文共4, 單字分佈共175 ??????
res = torch.argmax(predict, dim=1) #從distribution找最大機率作為輸出
middle_word = token_to_word[int(res)] #這裡將預測答案用vocab轉為單字
inputs.insert(2, middle_word) #這是加上中間預測詞的句子
print("ans : "," ".join(inputs))
