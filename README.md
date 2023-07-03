# word vector - CBOW practice
<CBOW所需>
1. 基本處理:將input轉成list of numbers(index), vocab dictionary, 句首句尾要標記(if pretrained) -> return vocab, tokens
2. dataset處理:放入模型的資料轉成cbow的(上下文, 中心詞)模式 -> class CbowDataset(Dataset)
3. 模型製作CbowModel(Module)
4. 訓練模型、檢測 
5. testing 測試

(1) vocab and corpus(dict index), include bos/eos token to index
(2) make cbow dataset
(3) Cbow model
(4) training Data
(5) training Model
(6) save "word vector"
(7) testing
