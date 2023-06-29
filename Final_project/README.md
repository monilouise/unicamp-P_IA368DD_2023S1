## ia368dd_projeto_final

**Forks:**

**1) Alteração no dataloader para utilizar os triplets ids:**

https://github.com/leobavila/splade
  * [train.py para train_from_triplets_ids.py](https://github.com/leobavila/splade/blob/main/splade/train_from_triplets_ids.py)
  * [DataLoaderWrapper para DataLoaderWrapperTripletsIds](https://github.com/leobavila/splade/blob/main/splade/datasets/dataloaders.py)
  * [SiamesePairsDataLoader para SiamesePairsDataLoaderTripletsIds](https://github.com/leobavila/splade/blob/main/splade/datasets/dataloaders.py)
    
**2) Alteração para utilizar o encoder do T5 (PTT5-V2):**

https://github.com/monilouise/splade
   * [Adição do mapeamento da classe SpladeT5](https://github.com/monilouise/splade/blob/main/splade/models/models_utils.py)
   * [Construção da classe SpladeT5](https://github.com/monilouise/splade/blob/main/splade/models/transformer_rep.py)
   
