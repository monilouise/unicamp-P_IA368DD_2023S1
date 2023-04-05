# T5 + doc2query - T5 Finetuning Experiments

Author: Monique Monteiro (moniquelouise@gmail.com)

## Dataset download


```python
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

```

    Mounted at /content/gdrive



```python
main_dir = "/content/gdrive/MyDrive/Unicamp-aula-6-3"
```


```python
!ls {main_dir}
```

    doc2query  msmarco_triples.train.tiny.tsv



```python
!wget https://storage.googleapis.com/unicamp-dl/ia368dd_2023s1/msmarco/msmarco_triples.train.tiny.tsv
```

    --2023-04-03 04:50:35--  https://storage.googleapis.com/unicamp-dl/ia368dd_2023s1/msmarco/msmarco_triples.train.tiny.tsv
    Resolving storage.googleapis.com (storage.googleapis.com)... 74.125.24.128, 142.251.10.128, 142.251.12.128, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|74.125.24.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 8076179 (7.7M) [text/tab-separated-values]
    Saving to: ‘msmarco_triples.train.tiny.tsv’
    
    msmarco_triples.tra 100%[===================>]   7.70M  7.37MB/s    in 1.0s    
    
    2023-04-03 04:50:37 (7.37 MB/s) - ‘msmarco_triples.train.tiny.tsv’ saved [8076179/8076179]
    



```python
!mv msmarco_triples.train.tiny.tsv {main_dir}
```

## Libraries installation


```python
import locale
locale.getpreferredencoding = lambda: "UTF-8"
```


```python
!pip install transformers
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting transformers
      Downloading transformers-4.27.4-py3-none-any.whl (6.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.8/6.8 MB[0m [31m89.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting tokenizers!=0.11.3,<0.14,>=0.11.1
      Downloading tokenizers-0.13.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (7.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.6/7.6 MB[0m [31m98.2 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.9/dist-packages (from transformers) (1.22.4)
    Requirement already satisfied: requests in /usr/local/lib/python3.9/dist-packages (from transformers) (2.27.1)
    Requirement already satisfied: filelock in /usr/local/lib/python3.9/dist-packages (from transformers) (3.10.7)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.9/dist-packages (from transformers) (2022.10.31)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.9/dist-packages (from transformers) (4.65.0)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.9/dist-packages (from transformers) (23.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.9/dist-packages (from transformers) (6.0)
    Collecting huggingface-hub<1.0,>=0.11.0
      Downloading huggingface_hub-0.13.3-py3-none-any.whl (199 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m199.8/199.8 KB[0m [31m23.2 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.9/dist-packages (from huggingface-hub<1.0,>=0.11.0->transformers) (4.5.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.9/dist-packages (from requests->transformers) (2022.12.7)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.9/dist-packages (from requests->transformers) (2.0.12)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.9/dist-packages (from requests->transformers) (1.26.15)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.9/dist-packages (from requests->transformers) (3.4)
    Installing collected packages: tokenizers, huggingface-hub, transformers
    Successfully installed huggingface-hub-0.13.3 tokenizers-0.13.2 transformers-4.27.4



```python
!pip install sentencepiece
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting sentencepiece
      Downloading sentencepiece-0.1.97-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m58.7 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: sentencepiece
    Successfully installed sentencepiece-0.1.97



```python
!pip install sacrebleu
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting sacrebleu
      Downloading sacrebleu-2.3.1-py3-none-any.whl (118 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m118.9/118.9 KB[0m [31m12.4 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: regex in /usr/local/lib/python3.9/dist-packages (from sacrebleu) (2022.10.31)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.9/dist-packages (from sacrebleu) (1.22.4)
    Collecting portalocker
      Downloading portalocker-2.7.0-py2.py3-none-any.whl (15 kB)
    Requirement already satisfied: lxml in /usr/local/lib/python3.9/dist-packages (from sacrebleu) (4.9.2)
    Collecting colorama
      Downloading colorama-0.4.6-py2.py3-none-any.whl (25 kB)
    Requirement already satisfied: tabulate>=0.8.9 in /usr/local/lib/python3.9/dist-packages (from sacrebleu) (0.8.10)
    Installing collected packages: portalocker, colorama, sacrebleu
    Successfully installed colorama-0.4.6 portalocker-2.7.0 sacrebleu-2.3.1



```python
!pip install datasets
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting datasets
      Downloading datasets-2.11.0-py3-none-any.whl (468 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m468.7/468.7 KB[0m [31m36.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.9/dist-packages (from datasets) (6.0)
    Requirement already satisfied: pyarrow>=8.0.0 in /usr/local/lib/python3.9/dist-packages (from datasets) (9.0.0)
    Collecting responses<0.19
      Downloading responses-0.18.0-py3-none-any.whl (38 kB)
    Collecting aiohttp
      Downloading aiohttp-3.8.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.0/1.0 MB[0m [31m53.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.9/dist-packages (from datasets) (2.27.1)
    Collecting dill<0.3.7,>=0.3.0
      Downloading dill-0.3.6-py3-none-any.whl (110 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m110.5/110.5 KB[0m [31m16.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting multiprocess
      Downloading multiprocess-0.70.14-py39-none-any.whl (132 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m132.9/132.9 KB[0m [31m18.7 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: pandas in /usr/local/lib/python3.9/dist-packages (from datasets) (1.4.4)
    Requirement already satisfied: huggingface-hub<1.0.0,>=0.11.0 in /usr/local/lib/python3.9/dist-packages (from datasets) (0.13.3)
    Requirement already satisfied: tqdm>=4.62.1 in /usr/local/lib/python3.9/dist-packages (from datasets) (4.65.0)
    Requirement already satisfied: fsspec[http]>=2021.11.1 in /usr/local/lib/python3.9/dist-packages (from datasets) (2023.3.0)
    Collecting xxhash
      Downloading xxhash-3.2.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (212 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m212.2/212.2 KB[0m [31m24.5 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.9/dist-packages (from datasets) (1.22.4)
    Requirement already satisfied: packaging in /usr/local/lib/python3.9/dist-packages (from datasets) (23.0)
    Collecting yarl<2.0,>=1.0
      Downloading yarl-1.8.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (264 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m264.6/264.6 KB[0m [31m35.2 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.9/dist-packages (from aiohttp->datasets) (22.2.0)
    Collecting aiosignal>=1.1.2
      Downloading aiosignal-1.3.1-py3-none-any.whl (7.6 kB)
    Collecting async-timeout<5.0,>=4.0.0a3
      Downloading async_timeout-4.0.2-py3-none-any.whl (5.8 kB)
    Collecting frozenlist>=1.1.1
      Downloading frozenlist-1.3.3-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (158 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m158.8/158.8 KB[0m [31m22.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: charset-normalizer<4.0,>=2.0 in /usr/local/lib/python3.9/dist-packages (from aiohttp->datasets) (2.0.12)
    Collecting multidict<7.0,>=4.5
      Downloading multidict-6.0.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (114 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m114.2/114.2 KB[0m [31m16.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: filelock in /usr/local/lib/python3.9/dist-packages (from huggingface-hub<1.0.0,>=0.11.0->datasets) (3.10.7)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.9/dist-packages (from huggingface-hub<1.0.0,>=0.11.0->datasets) (4.5.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.9/dist-packages (from requests>=2.19.0->datasets) (2022.12.7)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.9/dist-packages (from requests>=2.19.0->datasets) (3.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.9/dist-packages (from requests>=2.19.0->datasets) (1.26.15)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.9/dist-packages (from pandas->datasets) (2022.7.1)
    Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.9/dist-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.9/dist-packages (from python-dateutil>=2.8.1->pandas->datasets) (1.16.0)
    Installing collected packages: xxhash, multidict, frozenlist, dill, async-timeout, yarl, responses, multiprocess, aiosignal, aiohttp, datasets
    Successfully installed aiohttp-3.8.4 aiosignal-1.3.1 async-timeout-4.0.2 datasets-2.11.0 dill-0.3.6 frozenlist-1.3.3 multidict-6.0.4 multiprocess-0.70.14 responses-0.18.0 xxhash-3.2.0 yarl-1.8.2



```python
!pip install evaluate
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting evaluate
      Downloading evaluate-0.4.0-py3-none-any.whl (81 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m81.4/81.4 KB[0m [31m8.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: xxhash in /usr/local/lib/python3.9/dist-packages (from evaluate) (3.2.0)
    Requirement already satisfied: fsspec[http]>=2021.05.0 in /usr/local/lib/python3.9/dist-packages (from evaluate) (2023.3.0)
    Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.9/dist-packages (from evaluate) (2.27.1)
    Requirement already satisfied: tqdm>=4.62.1 in /usr/local/lib/python3.9/dist-packages (from evaluate) (4.65.0)
    Requirement already satisfied: pandas in /usr/local/lib/python3.9/dist-packages (from evaluate) (1.4.4)
    Requirement already satisfied: dill in /usr/local/lib/python3.9/dist-packages (from evaluate) (0.3.6)
    Requirement already satisfied: huggingface-hub>=0.7.0 in /usr/local/lib/python3.9/dist-packages (from evaluate) (0.13.3)
    Requirement already satisfied: datasets>=2.0.0 in /usr/local/lib/python3.9/dist-packages (from evaluate) (2.11.0)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.9/dist-packages (from evaluate) (1.22.4)
    Requirement already satisfied: responses<0.19 in /usr/local/lib/python3.9/dist-packages (from evaluate) (0.18.0)
    Requirement already satisfied: multiprocess in /usr/local/lib/python3.9/dist-packages (from evaluate) (0.70.14)
    Requirement already satisfied: packaging in /usr/local/lib/python3.9/dist-packages (from evaluate) (23.0)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.9/dist-packages (from datasets>=2.0.0->evaluate) (3.8.4)
    Requirement already satisfied: pyarrow>=8.0.0 in /usr/local/lib/python3.9/dist-packages (from datasets>=2.0.0->evaluate) (9.0.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.9/dist-packages (from datasets>=2.0.0->evaluate) (6.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.9/dist-packages (from huggingface-hub>=0.7.0->evaluate) (3.10.7)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.9/dist-packages (from huggingface-hub>=0.7.0->evaluate) (4.5.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.9/dist-packages (from requests>=2.19.0->evaluate) (3.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.9/dist-packages (from requests>=2.19.0->evaluate) (1.26.15)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.9/dist-packages (from requests>=2.19.0->evaluate) (2022.12.7)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.9/dist-packages (from requests>=2.19.0->evaluate) (2.0.12)
    Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.9/dist-packages (from pandas->evaluate) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.9/dist-packages (from pandas->evaluate) (2022.7.1)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.9/dist-packages (from aiohttp->datasets>=2.0.0->evaluate) (1.3.3)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.9/dist-packages (from aiohttp->datasets>=2.0.0->evaluate) (1.3.1)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.9/dist-packages (from aiohttp->datasets>=2.0.0->evaluate) (1.8.2)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.9/dist-packages (from aiohttp->datasets>=2.0.0->evaluate) (4.0.2)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.9/dist-packages (from aiohttp->datasets>=2.0.0->evaluate) (6.0.4)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.9/dist-packages (from aiohttp->datasets>=2.0.0->evaluate) (22.2.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.9/dist-packages (from python-dateutil>=2.8.1->pandas->evaluate) (1.16.0)
    Installing collected packages: evaluate
    Successfully installed evaluate-0.4.0


## Dataset creation


```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(f"{main_dir}/msmarco_triples.train.tiny.tsv", delimiter="\t", 
                 header=None, names=["query", "relevant_passage", "non_relevant_passage"])
X_train = df["relevant_passage"].tolist()
Y_train = df["query"].tolist()

#X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=1000, random_state=42)
```


```python
from torch.utils.data import Dataset, DataLoader

class Doc2QueryDataset(Dataset):
  def __init__(self, X, Y, tokenizer):
    self.tokenizer = tokenizer
    self.X = X
    self.Y = Y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):    
    tokenized_input = self.tokenizer(self.X[index])
    tokenized_query = self.tokenizer(self.Y[index])
    return {"input_ids": tokenized_input["input_ids"], 
            "attention_mask": tokenized_input["attention_mask"], 
            "labels": tokenized_query["input_ids"]}
    

```


```python
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AdamW, AutoTokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-base")

train_dataset = Doc2QueryDataset(X_train, Y_train, tokenizer)
val_dataset = Doc2QueryDataset(X_val, Y_val, tokenizer)

```


    Downloading (…)ve/main/spiece.model:   0%|          | 0.00/792k [00:00<?, ?B/s]



    Downloading (…)lve/main/config.json:   0%|          | 0.00/1.21k [00:00<?, ?B/s]


    /usr/local/lib/python3.9/dist-packages/transformers/models/t5/tokenization_t5.py:163: FutureWarning: This tokenizer was incorrectly instantiated with a model max length of 512 which will be corrected in Transformers v5.
    For now, this behavior is kept to avoid breaking backwards compatibility when padding/encoding with `truncation is True`.
    - Be aware that you SHOULD NOT rely on t5-base automatically truncating your input to 512 when padding/encoding.
    - If you want to encode/pad to sequences longer than 512 you can either instantiate this tokenizer with `model_max_length` or pass `max_length` when encoding/padding.
    - To avoid this warning, please instantiate this tokenizer with `model_max_length` set to your preferred value.
      warnings.warn(



```python
len(train_dataset)
```




    10000




```python
len(val_dataset)
```




    1000




```python
tokenizer.model_max_length
```




    512



## Metrics definition

Ref.: https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py


```python
import evaluate
```


```python
# Metric
metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    return result
```


    Downloading builder script:   0%|          | 0.00/8.15k [00:00<?, ?B/s]


## Finetuning Experiments


```python
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed
)
```

### 1st experiment (baseline): Default optimizer and learning rate scheduler


```python
#batch_size = 32 #bleu = 15.663058, lr default, split de validação igual a 0.2
#steps = 17
batch_size = 32 #bleu = 15.806152, lr default, split de validação igual a 0.2
steps = 50
epochs = 100
```


```python
import numpy as np

model = T5ForConditionalGeneration.from_pretrained("t5-base")


print("batch size = ", batch_size)
print("len train_dataset = ", len(train_dataset))

training_args = Seq2SeqTrainingArguments(output_dir=f"{main_dir}/doc2query",
                                          overwrite_output_dir=True,
                                          per_device_train_batch_size=batch_size,
                                          per_device_eval_batch_size=batch_size,
                                          gradient_accumulation_steps=8,
                                          evaluation_strategy='steps',
                                          eval_steps=steps, logging_steps=steps, 
                                          save_steps=steps,
                                          predict_with_generate=True,
                                          fp16=True, 
                                          num_train_epochs=epochs,
                                          load_best_model_at_end=True,
                                          metric_for_best_model='bleu',
                                          save_total_limit = 2
                                        )

#If you use mixed precision, you need all your tensors to have dimensions that are multiple of 8s to maximize the benefits of your tensor cores.
#So pas_to_multiple_of=8 is a good value
#Ref.: https://discuss.huggingface.co/t/whats-a-good-value-for-pad-to-multiple-of/1481

#Se não usar o collator e tokenizar com parâmetros além da entrada, todo tipo de erro acontece.
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8 if training_args.fp16 else None,
)

trainer = Seq2SeqTrainer(model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        data_collator=data_collator,
                        tokenizer=tokenizer,
                        compute_metrics=compute_metrics
                        )

train_results = trainer.train()

```

    batch size =  32
    len train_dataset =  10000


    /usr/local/lib/python3.9/dist-packages/transformers/optimization.py:391: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      warnings.warn(




    <div>

      <progress value='201' max='3900' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [ 201/3900 07:47 < 2:24:51, 0.43 it/s, Epoch 5.11/100]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Bleu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>50</td>
      <td>3.243800</td>
      <td>1.779949</td>
      <td>13.804612</td>
    </tr>
    <tr>
      <td>100</td>
      <td>1.946700</td>
      <td>1.633787</td>
      <td>16.332318</td>
    </tr>
    <tr>
      <td>150</td>
      <td>1.988800</td>
      <td>1.606424</td>
      <td>15.381401</td>
    </tr>
  </tbody>
</table><p>
    <div>

      <progress value='6' max='32' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [ 6/32 00:05 < 00:26, 0.98 it/s]
    </div>




    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-20-378eb99a9412> in <cell line: 46>()
         44                         )
         45 
    ---> 46 train_results = trainer.train()
    

    /usr/local/lib/python3.9/dist-packages/transformers/trainer.py in train(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
       1631             self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
       1632         )
    -> 1633         return inner_training_loop(
       1634             args=args,
       1635             resume_from_checkpoint=resume_from_checkpoint,


    /usr/local/lib/python3.9/dist-packages/transformers/trainer.py in _inner_training_loop(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
       1900                         tr_loss_step = self.training_step(model, inputs)
       1901                 else:
    -> 1902                     tr_loss_step = self.training_step(model, inputs)
       1903 
       1904                 if (


    /usr/local/lib/python3.9/dist-packages/transformers/trainer.py in _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval)
       2234                     )
       2235             else:
    -> 2236                 metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
       2237             self._report_to_hp_search(trial, self.state.global_step, metrics)
       2238 


    /usr/local/lib/python3.9/dist-packages/transformers/trainer_seq2seq.py in evaluate(self, eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)
         76         self._gen_kwargs = gen_kwargs
         77 
    ---> 78         return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
         79 
         80     def predict(


    /usr/local/lib/python3.9/dist-packages/transformers/trainer.py in evaluate(self, eval_dataset, ignore_keys, metric_key_prefix)
       2930 
       2931         eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
    -> 2932         output = eval_loop(
       2933             eval_dataloader,
       2934             description="Evaluation",


    /usr/local/lib/python3.9/dist-packages/transformers/trainer.py in evaluation_loop(self, dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
       3111 
       3112             # Prediction step
    -> 3113             loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
       3114             inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None
       3115 


    /usr/local/lib/python3.9/dist-packages/transformers/trainer_seq2seq.py in prediction_step(self, model, inputs, prediction_loss_only, ignore_keys)
        186         # users from preparing a dataset with `decoder_input_ids`.
        187         inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
    --> 188         generated_tokens = self.model.generate(**inputs, **gen_kwargs)
        189 
        190         # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop


    /usr/local/lib/python3.9/dist-packages/torch/autograd/grad_mode.py in decorate_context(*args, **kwargs)
         25         def decorate_context(*args, **kwargs):
         26             with self.clone():
    ---> 27                 return func(*args, **kwargs)
         28         return cast(F, decorate_context)
         29 


    /usr/local/lib/python3.9/dist-packages/transformers/generation/utils.py in generate(self, inputs, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, **kwargs)
       1404 
       1405             # 11. run greedy search
    -> 1406             return self.greedy_search(
       1407                 input_ids,
       1408                 logits_processor=logits_processor,


    /usr/local/lib/python3.9/dist-packages/transformers/generation/utils.py in greedy_search(self, input_ids, logits_processor, stopping_criteria, max_length, pad_token_id, eos_token_id, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, synced_gpus, **model_kwargs)
       2153         if isinstance(eos_token_id, int):
       2154             eos_token_id = [eos_token_id]
    -> 2155         eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
       2156         output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
       2157         output_attentions = (


    KeyboardInterrupt: 


### 2nd Experiment: AdamW and constant learning rate = 1e-4


```python
from transformers.optimization import get_constant_schedule
```


```python
#batch_size=8 #bleu=17.585528 com split=0.2
#batch_size=16 #bleu=16.439199 com split=0.2
batch_size=8
steps=50
epochs=100
```


```python
import numpy as np

model = T5ForConditionalGeneration.from_pretrained("t5-base")

optimizer = AdamW(model.parameters(), lr=1e-4)
lr_scheduler = get_constant_schedule(optimizer)
print("batch size = ", batch_size)
print("len train_dataset = ", len(train_dataset))

training_args = Seq2SeqTrainingArguments(output_dir=f"{main_dir}/doc2query",
                                          overwrite_output_dir=True,
                                          per_device_train_batch_size=batch_size,
                                          per_device_eval_batch_size=batch_size,
                                          gradient_accumulation_steps=8,
                                          evaluation_strategy='steps',
                                          eval_steps=steps, logging_steps=steps, 
                                          save_steps=steps,
                                          predict_with_generate=True,
                                          fp16=True, 
                                          num_train_epochs=epochs,
                                          load_best_model_at_end=True,
                                          metric_for_best_model='bleu',
                                          save_total_limit = 2
                                        )

#If you use mixed precision, you need all your tensors to have dimensions that are multiple of 8s to maximize the benefits of your tensor cores.
#So pas_to_multiple_of=8 is a good value
#Ref.: https://discuss.huggingface.co/t/whats-a-good-value-for-pad-to-multiple-of/1481

#Se não usar o collator e tokenizar com parâmetros além da entrada, todo tipo de erro acontece.
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8 if training_args.fp16 else None,
)

trainer = Seq2SeqTrainer(model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        data_collator=data_collator,
                        tokenizer=tokenizer,
                        compute_metrics=compute_metrics,
                        optimizers=(optimizer,lr_scheduler)
                        )

train_results = trainer.train()

```

    /usr/local/lib/python3.9/dist-packages/transformers/optimization.py:391: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      warnings.warn(


    batch size =  8
    len train_dataset =  10000




    <div>

      <progress value='203' max='15600' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [  203/15600 10:31 < 13:25:38, 0.32 it/s, Epoch 1.29/100]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Bleu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>50</td>
      <td>2.747800</td>
      <td>1.760042</td>
      <td>14.191379</td>
    </tr>
    <tr>
      <td>100</td>
      <td>1.936300</td>
      <td>1.634877</td>
      <td>16.028759</td>
    </tr>
    <tr>
      <td>150</td>
      <td>1.842900</td>
      <td>1.591470</td>
      <td>17.079439</td>
    </tr>
    <tr>
      <td>200</td>
      <td>1.670900</td>
      <td>1.576075</td>
      <td>17.013726</td>
    </tr>
  </tbody>
</table><p>



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-27-8ac408d44f91> in <cell line: 48>()
         46                         )
         47 
    ---> 48 train_results = trainer.train()
    

    /usr/local/lib/python3.9/dist-packages/transformers/trainer.py in train(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
       1631             self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
       1632         )
    -> 1633         return inner_training_loop(
       1634             args=args,
       1635             resume_from_checkpoint=resume_from_checkpoint,


    /usr/local/lib/python3.9/dist-packages/transformers/trainer.py in _inner_training_loop(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
       1900                         tr_loss_step = self.training_step(model, inputs)
       1901                 else:
    -> 1902                     tr_loss_step = self.training_step(model, inputs)
       1903 
       1904                 if (


    /usr/local/lib/python3.9/dist-packages/transformers/trainer_callback.py in on_step_end(self, args, state, control)
        373 
        374     def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
    --> 375         return self.call_event("on_step_end", args, state, control)
        376 
        377     def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):


    /usr/local/lib/python3.9/dist-packages/transformers/trainer_callback.py in call_event(self, event, args, state, control, **kwargs)
        395     def call_event(self, event, args, state, control, **kwargs):
        396         for callback in self.callbacks:
    --> 397             result = getattr(callback, event)(
        398                 args,
        399                 state,


    /usr/local/lib/python3.9/dist-packages/transformers/utils/notebook.py in on_step_end(self, args, state, control, **kwargs)
        287     def on_step_end(self, args, state, control, **kwargs):
        288         epoch = int(state.epoch) if int(state.epoch) == state.epoch else f"{state.epoch:.2f}"
    --> 289         self.training_tracker.update(
        290             state.global_step + 1,
        291             comment=f"Epoch {epoch}/{state.num_train_epochs}",


    /usr/local/lib/python3.9/dist-packages/transformers/utils/notebook.py in update(self, value, force_update, comment)
        159             elif self.average_time_per_item is not None:
        160                 self.predicted_remaining = self.average_time_per_item * (self.total - value)
    --> 161             self.update_bar(value)
        162             self.last_value = value
        163             self.last_time = current_time


    /usr/local/lib/python3.9/dist-packages/transformers/utils/notebook.py in update_bar(self, value, comment)
        180             self.label += f", {1/self.average_time_per_item:.2f} it/s"
        181         self.label += "]" if self.comment is None or len(self.comment) == 0 else f", {self.comment}]"
    --> 182         self.display()
        183 
        184     def display(self):


    /usr/local/lib/python3.9/dist-packages/transformers/utils/notebook.py in display(self)
        223             self.output = disp.display(disp.HTML(self.html_code), display_id=True)
        224         else:
    --> 225             self.output.update(disp.HTML(self.html_code))
        226 
        227     def write_line(self, values):


    /usr/local/lib/python3.9/dist-packages/IPython/core/display.py in update(self, obj, **kwargs)
        398             additional keyword arguments passed to update_display
        399         """
    --> 400         update_display(obj, display_id=self.display_id, **kwargs)
        401 
        402 


    /usr/local/lib/python3.9/dist-packages/IPython/core/display.py in update_display(obj, display_id, **kwargs)
        348     """
        349     kwargs['update'] = True
    --> 350     display(obj, display_id=display_id, **kwargs)
        351 
        352 


    /usr/local/lib/python3.9/dist-packages/IPython/core/display.py in display(include, exclude, metadata, transient, display_id, *objs, **kwargs)
        325                 # kwarg-specified metadata gets precedence
        326                 _merge(md_dict, metadata)
    --> 327             publish_display_data(data=format_dict, metadata=md_dict, **kwargs)
        328     if display_id:
        329         return DisplayHandle(display_id)


    /usr/local/lib/python3.9/dist-packages/IPython/core/display.py in publish_display_data(data, metadata, source, transient, **kwargs)
        117         kwargs['transient'] = transient
        118 
    --> 119     display_pub.publish(
        120         data=data,
        121         metadata=metadata,


    /usr/local/lib/python3.9/dist-packages/ipykernel/zmqshell.py in publish(self, data, metadata, source, transient, update)
        113             If True, send an update_display_data message instead of display_data.
        114         """
    --> 115         self._flush_streams()
        116         if metadata is None:
        117             metadata = {}


    /usr/local/lib/python3.9/dist-packages/ipykernel/zmqshell.py in _flush_streams(self)
         80     def _flush_streams(self):
         81         """flush IO Streams prior to display"""
    ---> 82         sys.stdout.flush()
         83         sys.stderr.flush()
         84 


    /usr/local/lib/python3.9/dist-packages/ipykernel/iostream.py in flush(self)
        348                 self.pub_thread.schedule(evt.set)
        349                 # and give a timeout to avoid
    --> 350                 if not evt.wait(self.flush_timeout):
        351                     # write directly to __stderr__ instead of warning because
        352                     # if this is happening sys.stderr may be the problem.


    /usr/lib/python3.9/threading.py in wait(self, timeout)
        579             signaled = self._flag
        580             if not signaled:
    --> 581                 signaled = self._cond.wait(timeout)
        582             return signaled
        583 


    /usr/lib/python3.9/threading.py in wait(self, timeout)
        314             else:
        315                 if timeout > 0:
    --> 316                     gotit = waiter.acquire(True, timeout)
        317                 else:
        318                     gotit = waiter.acquire(False)


    KeyboardInterrupt: 



```python
trainer.save_model()
```

### 3rd Experiment: AdaFactor optimizer



```python
batch_size=32
#batch_size=8
steps=50
#epochs=100
epochs=13
```

Without FP16, the model achieved a high BLEU value (>22), but the validation loss increased as BLEU also increased (!).  On the other hand, with FP16, the model gets stuck at BLEU=11 and does not improve for several iterations.  

After replacing fp16 by bf16, the same "overfiting-like" behaviour occurred.  So I saved the model with early stoping, up to a point with minimal validation loss and maximum BLEU (19.92) ("normal behaviour). 




```python
import numpy as np
from transformers.optimization import Adafactor, AdafactorSchedule

model = T5ForConditionalGeneration.from_pretrained("t5-base")

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True)
lr_scheduler = AdafactorSchedule(optimizer)

print("batch size = ", batch_size)
print("len train_dataset = ", len(train_dataset))

training_args = Seq2SeqTrainingArguments(output_dir=f"{main_dir}/doc2query",
                                          overwrite_output_dir=True,
                                          per_device_train_batch_size=batch_size,
                                          per_device_eval_batch_size=batch_size,
                                          gradient_accumulation_steps=8,
                                          evaluation_strategy='steps',
                                          eval_steps=steps, logging_steps=steps, 
                                          save_steps=steps,
                                          predict_with_generate=True,
                                          #fp16=True,
                                          bf16=True,
                                          num_train_epochs=epochs,
                                          load_best_model_at_end=True,
                                          metric_for_best_model='bleu',
                                          save_total_limit = 2
                                          
                                        )

#If you use mixed precision, you need all your tensors to have dimensions that are multiple of 8s to maximize the benefits of your tensor cores.
#So pas_to_multiple_of=8 is a good value
#Ref.: https://discuss.huggingface.co/t/whats-a-good-value-for-pad-to-multiple-of/1481

#Se não usar o collator e tokenizar com parâmetros além da entrada, todo tipo de erro acontece.
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8 if training_args.fp16 else None,
)

trainer = Seq2SeqTrainer(model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        data_collator=data_collator,
                        tokenizer=tokenizer,
                        compute_metrics=compute_metrics,
                        optimizers=(optimizer, lr_scheduler)
                        )

train_results = trainer.train()

```

    batch size =  32
    len train_dataset =  10000




    <div>

      <progress value='507' max='507' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [507/507 19:49, Epoch 12/13]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Bleu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>50</td>
      <td>5.125100</td>
      <td>4.091910</td>
      <td>1.297463</td>
    </tr>
    <tr>
      <td>100</td>
      <td>3.026000</td>
      <td>1.969804</td>
      <td>8.736579</td>
    </tr>
    <tr>
      <td>150</td>
      <td>2.004700</td>
      <td>1.700381</td>
      <td>14.301186</td>
    </tr>
    <tr>
      <td>200</td>
      <td>1.807800</td>
      <td>1.611367</td>
      <td>15.833664</td>
    </tr>
    <tr>
      <td>250</td>
      <td>1.684800</td>
      <td>1.558003</td>
      <td>17.080495</td>
    </tr>
    <tr>
      <td>300</td>
      <td>1.590700</td>
      <td>1.520123</td>
      <td>18.581309</td>
    </tr>
    <tr>
      <td>350</td>
      <td>1.496000</td>
      <td>1.498664</td>
      <td>19.053267</td>
    </tr>
    <tr>
      <td>400</td>
      <td>1.417800</td>
      <td>1.487062</td>
      <td>18.643619</td>
    </tr>
    <tr>
      <td>450</td>
      <td>1.330200</td>
      <td>1.479391</td>
      <td>19.920438</td>
    </tr>
    <tr>
      <td>500</td>
      <td>1.255900</td>
      <td>1.480590</td>
      <td>19.761084</td>
    </tr>
  </tbody>
</table><p>



```python
# See https://stats.stackexchange.com/questions/282160/how-is-it-possible-that-validation-loss-is-increasing-while-validation-accuracy and https://forum.opennmt.net/t/scorer-test-set-vs-validation-set/4517/3
```


```python
trainer.save_model()
```


```python
metrics = trainer.evaluate()
```



<div>

  <progress value='32' max='32' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [32/32 00:18]
</div>




```python
metrics
```




    {'eval_loss': 1.479391098022461,
     'eval_bleu': 19.920438361367808,
     'eval_runtime': 35.2199,
     'eval_samples_per_second': 28.393,
     'eval_steps_per_second': 0.909,
     'epoch': 12.96}




```python
import json

with open(f"{main_dir}/doc2query/metrics.json", 'w') as f:
  json.dump(metrics,f)
```

### 4th Experiment - mixing different batch sizes

Inspired by PALM paper
It's based on starting the training with a low batch size and latter incresase this batch size.


```python
import os
```


```python
steps=50
```


```python
import numpy as np

model = T5ForConditionalGeneration.from_pretrained("t5-base")

def train(model, batch_size, epochs=3):
  print('batch size = ', batch_size)
  training_args = Seq2SeqTrainingArguments(output_dir=f"{main_dir}/doc2query",
                                            overwrite_output_dir=True,
                                            per_device_train_batch_size=batch_size,
                                            per_device_eval_batch_size=batch_size,
                                            gradient_accumulation_steps=8,
                                            evaluation_strategy='steps',
                                            eval_steps=steps, logging_steps=steps, 
                                            save_steps=steps,
                                            predict_with_generate=True,
                                            fp16=True, 
                                            num_train_epochs=epochs,
                                            load_best_model_at_end=True,
                                            metric_for_best_model='bleu',
                                            save_total_limit = 2
                                          )

  #If you use mixed precision, you need all your tensors to have dimensions that are multiple of 8s to maximize the benefits of your tensor cores.
  #So pas_to_multiple_of=8 is a good value
  #Ref.: https://discuss.huggingface.co/t/whats-a-good-value-for-pad-to-multiple-of/1481

  #Se não usar o collator e tokenizar com parâmetros além da entrada, todo tipo de erro acontece.
  data_collator = DataCollatorForSeq2Seq( 
      tokenizer,
      model=model,
      label_pad_token_id=-100,
      pad_to_multiple_of=8 if training_args.fp16 else None,
  )

  trainer = Seq2SeqTrainer(model=model,
                          args=training_args,
                          train_dataset=train_dataset,
                          eval_dataset=val_dataset,
                          data_collator=data_collator,
                          tokenizer=tokenizer,
                          compute_metrics=compute_metrics
                          )

  train_results = trainer.train()
  return trainer


```


    Downloading pytorch_model.bin:   0%|          | 0.00/892M [00:00<?, ?B/s]



    Downloading (…)neration_config.json:   0%|          | 0.00/147 [00:00<?, ?B/s]



```python
trainer = train(model, 8)
trainer = train(model, 32, 12)
```


```python
trainer.save_model()
metrics = trainer.evaluate()

```


```python
import json

with open(f"{main_dir}/doc2query/metrics.json", 'w') as f:
  json.dump(metrics,f)
```


```python
import os

os.rename(f"{main_dir}/doc2query", f"{main_dir}/doc2query_default_b8-b32")
```

Now the opposite - start with a high batch size and decrease it later.


```python
model = T5ForConditionalGeneration.from_pretrained("t5-base")
trainer = train(model, 32, 12)
trainer = train(model, 8)
```

    batch size =  32


    /usr/local/lib/python3.9/dist-packages/transformers/optimization.py:391: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      warnings.warn(




    <div>

      <progress value='468' max='468' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [468/468 17:29, Epoch 11/12]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Bleu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>50</td>
      <td>3.188800</td>
      <td>1.774593</td>
      <td>13.636776</td>
    </tr>
    <tr>
      <td>100</td>
      <td>2.319900</td>
      <td>1.658584</td>
      <td>14.665341</td>
    </tr>
    <tr>
      <td>150</td>
      <td>2.147500</td>
      <td>1.663010</td>
      <td>13.621872</td>
    </tr>
    <tr>
      <td>200</td>
      <td>1.919200</td>
      <td>1.661236</td>
      <td>13.445569</td>
    </tr>
    <tr>
      <td>250</td>
      <td>1.980200</td>
      <td>1.659800</td>
      <td>13.465747</td>
    </tr>
    <tr>
      <td>300</td>
      <td>2.086200</td>
      <td>1.659727</td>
      <td>13.465747</td>
    </tr>
    <tr>
      <td>350</td>
      <td>2.010900</td>
      <td>1.659710</td>
      <td>13.465747</td>
    </tr>
    <tr>
      <td>400</td>
      <td>1.982900</td>
      <td>1.659710</td>
      <td>13.465747</td>
    </tr>
    <tr>
      <td>450</td>
      <td>2.055800</td>
      <td>1.659710</td>
      <td>13.465747</td>
    </tr>
  </tbody>
</table><p>


    batch size =  8


    /usr/local/lib/python3.9/dist-packages/transformers/optimization.py:391: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      warnings.warn(




    <div>

      <progress value='468' max='468' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [468/468 20:23, Epoch 2/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Bleu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>50</td>
      <td>1.815800</td>
      <td>1.624490</td>
      <td>15.565391</td>
    </tr>
    <tr>
      <td>100</td>
      <td>1.786700</td>
      <td>1.597662</td>
      <td>16.743959</td>
    </tr>
    <tr>
      <td>150</td>
      <td>1.809300</td>
      <td>1.583537</td>
      <td>16.668995</td>
    </tr>
    <tr>
      <td>200</td>
      <td>1.678000</td>
      <td>1.575306</td>
      <td>16.547084</td>
    </tr>
    <tr>
      <td>250</td>
      <td>1.800400</td>
      <td>1.563522</td>
      <td>17.304830</td>
    </tr>
    <tr>
      <td>300</td>
      <td>1.700900</td>
      <td>1.555191</td>
      <td>16.679671</td>
    </tr>
    <tr>
      <td>350</td>
      <td>1.575900</td>
      <td>1.551357</td>
      <td>17.205937</td>
    </tr>
    <tr>
      <td>400</td>
      <td>1.805400</td>
      <td>1.548646</td>
      <td>16.418515</td>
    </tr>
    <tr>
      <td>450</td>
      <td>1.667300</td>
      <td>1.548901</td>
      <td>16.356012</td>
    </tr>
  </tbody>
</table><p>



```python
trainer.save_model()
metrics = trainer.evaluate()
with open(f"{main_dir}/doc2query/metrics.json", 'w') as f:
  json.dump(metrics,f)
```


```python
os.rename(f"{main_dir}/doc2query", f"{main_dir}/doc2query_default_b32-b8")
```

Finally, for the best result (smaller to higher), increase progressively from 8 to 16 then 32.


```python
model = T5ForConditionalGeneration.from_pretrained("t5-base")
trainer = train(model, 8)
trainer = train(model, 16, 6)
trainer = train(model, 32, 12)

```

    batch size =  8


    /usr/local/lib/python3.9/dist-packages/transformers/optimization.py:391: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      warnings.warn(




    <div>

      <progress value='451' max='468' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [451/468 17:49 < 00:40, 0.42 it/s, Epoch 2.88/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Bleu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>50</td>
      <td>2.980300</td>
      <td>1.846013</td>
      <td>11.816433</td>
    </tr>
    <tr>
      <td>100</td>
      <td>2.002600</td>
      <td>1.707517</td>
      <td>14.135295</td>
    </tr>
    <tr>
      <td>150</td>
      <td>1.962800</td>
      <td>1.661441</td>
      <td>14.941337</td>
    </tr>
    <tr>
      <td>200</td>
      <td>1.773800</td>
      <td>1.631891</td>
      <td>14.921651</td>
    </tr>
    <tr>
      <td>250</td>
      <td>1.808900</td>
      <td>1.613374</td>
      <td>16.065293</td>
    </tr>
    <tr>
      <td>300</td>
      <td>1.798500</td>
      <td>1.602750</td>
      <td>16.259420</td>
    </tr>
    <tr>
      <td>350</td>
      <td>1.666100</td>
      <td>1.596123</td>
      <td>16.504767</td>
    </tr>
    <tr>
      <td>400</td>
      <td>1.706400</td>
      <td>1.589016</td>
      <td>16.112679</td>
    </tr>
  </tbody>
</table><p>
    <div>

      <progress value='106' max='125' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [106/125 00:34 < 00:06, 3.03 it/s]
    </div>





    <div>

      <progress value='468' max='468' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [468/468 19:14, Epoch 2/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Bleu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>50</td>
      <td>2.980300</td>
      <td>1.846013</td>
      <td>11.816433</td>
    </tr>
    <tr>
      <td>100</td>
      <td>2.002600</td>
      <td>1.707517</td>
      <td>14.135295</td>
    </tr>
    <tr>
      <td>150</td>
      <td>1.962800</td>
      <td>1.661441</td>
      <td>14.941337</td>
    </tr>
    <tr>
      <td>200</td>
      <td>1.773800</td>
      <td>1.631891</td>
      <td>14.921651</td>
    </tr>
    <tr>
      <td>250</td>
      <td>1.808900</td>
      <td>1.613374</td>
      <td>16.065293</td>
    </tr>
    <tr>
      <td>300</td>
      <td>1.798500</td>
      <td>1.602750</td>
      <td>16.259420</td>
    </tr>
    <tr>
      <td>350</td>
      <td>1.666100</td>
      <td>1.596123</td>
      <td>16.504767</td>
    </tr>
    <tr>
      <td>400</td>
      <td>1.706400</td>
      <td>1.589016</td>
      <td>16.112679</td>
    </tr>
    <tr>
      <td>450</td>
      <td>1.699900</td>
      <td>1.587041</td>
      <td>16.143037</td>
    </tr>
  </tbody>
</table><p>


    batch size =  16


    /usr/local/lib/python3.9/dist-packages/transformers/optimization.py:391: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      warnings.warn(




    <div>

      <progress value='468' max='468' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [468/468 15:41, Epoch 5/6]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Bleu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>50</td>
      <td>1.770400</td>
      <td>1.572804</td>
      <td>16.613320</td>
    </tr>
    <tr>
      <td>100</td>
      <td>1.609600</td>
      <td>1.559006</td>
      <td>17.371010</td>
    </tr>
    <tr>
      <td>150</td>
      <td>1.599300</td>
      <td>1.543522</td>
      <td>18.069003</td>
    </tr>
    <tr>
      <td>200</td>
      <td>1.552400</td>
      <td>1.536495</td>
      <td>18.468013</td>
    </tr>
    <tr>
      <td>250</td>
      <td>1.629900</td>
      <td>1.533385</td>
      <td>18.725323</td>
    </tr>
    <tr>
      <td>300</td>
      <td>1.503700</td>
      <td>1.527405</td>
      <td>18.090745</td>
    </tr>
    <tr>
      <td>350</td>
      <td>1.531800</td>
      <td>1.524894</td>
      <td>18.666940</td>
    </tr>
    <tr>
      <td>400</td>
      <td>1.461900</td>
      <td>1.524966</td>
      <td>18.662090</td>
    </tr>
    <tr>
      <td>450</td>
      <td>1.443100</td>
      <td>1.523645</td>
      <td>18.312985</td>
    </tr>
  </tbody>
</table><p>


    batch size =  32




    <div>

      <progress value='468' max='468' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [468/468 16:17, Epoch 11/12]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Bleu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>50</td>
      <td>1.502200</td>
      <td>1.529532</td>
      <td>18.970029</td>
    </tr>
    <tr>
      <td>100</td>
      <td>1.454100</td>
      <td>1.522025</td>
      <td>19.221423</td>
    </tr>
    <tr>
      <td>150</td>
      <td>1.514900</td>
      <td>1.518564</td>
      <td>18.875055</td>
    </tr>
    <tr>
      <td>200</td>
      <td>1.485900</td>
      <td>1.517217</td>
      <td>18.963882</td>
    </tr>
    <tr>
      <td>250</td>
      <td>1.405100</td>
      <td>1.495193</td>
      <td>19.531806</td>
    </tr>
    <tr>
      <td>300</td>
      <td>1.391100</td>
      <td>1.505102</td>
      <td>18.038478</td>
    </tr>
    <tr>
      <td>350</td>
      <td>1.449400</td>
      <td>1.518167</td>
      <td>17.902955</td>
    </tr>
    <tr>
      <td>400</td>
      <td>1.413400</td>
      <td>1.517734</td>
      <td>17.793682</td>
    </tr>
    <tr>
      <td>450</td>
      <td>1.438000</td>
      <td>1.517273</td>
      <td>17.812849</td>
    </tr>
  </tbody>
</table><p>



```python
trainer.save_model()
metrics = trainer.evaluate()
with open(f"{main_dir}/doc2query/metrics.json", 'w') as f:
  json.dump(metrics,f)
```



<div>

  <progress value='32' max='32' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [32/32 00:17]
</div>




```python
os.rename(f"{main_dir}/doc2query", f"{main_dir}/doc2query_default_b8-b16-b32")
```

## Conclusions for the next step

Use 3 candidate models to expand the queries:

1.   doc2query-adafactor-bs-32-split-1000-no-fp16 (BLEU = 22.46)
2.   doc2query-adafactor-bs-32-split-1000-withbf16-early-stoping (BLEU = 19.92)
3.   doc2query_default_b8-b16-b32 (BLEU = 19.53, with expected behavior)



```python

```
