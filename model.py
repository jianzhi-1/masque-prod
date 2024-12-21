import torch
import torch.nn as nn

device = torch.device("cuda")

class EmotionModel(nn.Module):
    def transform(self, batch):
        raise NotImplementedError()

    def compute_loss(self, batch):
        batch_size, seq_length, mels_dim = batch["ai_sparc"].shape
        assert batch["data_sparc"].shape == (batch_size, seq_length, mels_dim)

        predicted_mel = self.transform(batch)
        assert predicted_mel.shape == (batch_size, seq_length, mels_dim)

        target_mel = batch["data_sparc"]
        assert target_mel.shape == (batch_size, seq_length, mels_dim)

        assert mels_dim == 14 # SPARC only has 14 dimensions
        loss = torch.sum((predicted_mel - target_mel)**2)
        return loss
  
    def get_validation_metric(self, validation_dataset, batch_size=64):
        dataset = validation_dataset # replace because of caching efficiency
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=dataset.collate
        )
        self.eval()
        total_mse = 0.0
        total = 0
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                loss = self.compute_loss(batch)
                total_mse += loss
                total += batch["ai_sparc"].size(0)

        return total_mse/total

class AddPositionalEncoding(nn.Module):
    def __init__(self, d_model=256, input_dropout=0.1, timing_dropout=0.1, max_len=2048):
        super().__init__()
        self.max_len = max_len
        self.timing_table = nn.Parameter(torch.zeros(max_len))
        nn.init.normal_(self.timing_table)
        self.input_dropout = nn.Dropout(input_dropout)
        self.timing_dropout = nn.Dropout(timing_dropout)

    def forward(self, x, mask):
        batch_size, seq_length, d_model = x.shape
        assert x.shape == (batch_size, seq_length, d_model)
        assert mask.shape == (batch_size, seq_length)
        assert seq_length < self.max_len
        x = self.input_dropout(x)
        timing = self.timing_table[:seq_length]
        timing = self.timing_dropout(timing)
        assert timing.shape == (seq_length,), f"{timing.shape}"
        assert timing.unsqueeze(0).unsqueeze(2).shape == (1, seq_length, 1), f"{timing.unsqueeze(0).unsqueeze(2).shape}"
        assert (x + timing.unsqueeze(0).unsqueeze(2)).shape == (batch_size, seq_length, d_model), f"{(x + timing.unsqueeze(0).unsqueeze(2)).shape}"
        assert mask.unsqueeze(-1).expand(-1, -1, d_model).shape == (batch_size, seq_length, d_model), f"{mask.unsqueeze(-1).expand(-1, -1, d_model)}"
        return torch.where(mask.unsqueeze(-1).expand(-1, -1, d_model)==False, x + timing.unsqueeze(0).unsqueeze(2), x)

class TransformerEmotionModel(EmotionModel):
    def __init__(self, d_model=512, num_encoder_layers=6, dropout=0.1):
        super().__init__()
        self.n_mels = 14 # SPARC only has 14 features
        self.d_model = d_model
        self.add_timing = AddPositionalEncoding(d_model)
        self.num_encoder_layers = num_encoder_layers
        encoder_ls = []
        for _ in range(num_encoder_layers):
            encoder_ls.append(nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, norm_first=False, dropout=dropout, dim_feedforward=d_model))
        self.encoder_layers = nn.ModuleList(encoder_ls)
        self.embedding_layer = nn.Embedding(11, d_model) # len(self.label_encoder) = 11
        self.pre_projection_layer = nn.Linear(self.n_mels, d_model)
        self.post_projection_layer = nn.Linear(d_model, self.n_mels)

    def transform(self, batch):
        
        batch_size, seq_length, _ = batch["ai_sparc"].shape
        assert batch["ai_sparc"].shape == (batch_size, seq_length, self.n_mels)
        
        batch_input = batch["ai_sparc"]
        assert batch_input.shape == (batch_size, seq_length, self.n_mels)
        assert not torch.any(torch.isnan(batch_input))

        label = batch["labels"]
        mask = batch["mask"]
        
        assert mask.shape == (batch_size, seq_length)
        assert not torch.any(torch.isnan(mask))
        mask = torch.cat((torch.full((batch_size, 1), False).to(device), mask), 1)
        assert mask.shape == (batch_size, 1 + seq_length)
        
        assert label.shape == (batch_size,)
        label_embedded = self.embedding_layer(label).unsqueeze(1)
        assert label_embedded.shape == (batch_size, 1, self.d_model)
        assert not torch.any(torch.isnan(label_embedded))
        
        pre_adjoined = self.pre_projection_layer(batch_input)
        assert pre_adjoined.shape == (batch_size, seq_length, self.d_model)
        assert not torch.any(torch.isnan(pre_adjoined))
        
        adjoined = torch.cat((label_embedded, pre_adjoined), 1)
        assert adjoined.shape == (batch_size, 1 + seq_length, self.d_model)
        assert not torch.any(torch.isnan(adjoined))
        
        adjoined_with_timing = self.add_timing(adjoined, mask)
        assert adjoined_with_timing.shape == (batch_size, 1 + seq_length, self.d_model)
        assert not torch.any(torch.isnan(adjoined_with_timing))
        
        after_encoder = adjoined_with_timing
        
        for i in range(self.num_encoder_layers):
            after_encoder = self.encoder_layers[i](after_encoder, src_key_padding_mask=mask)
            assert after_encoder.shape == (batch_size, 1 + seq_length, self.d_model)
            assert not torch.any(torch.isnan(after_encoder))
        
        post_adjoined = self.post_projection_layer(after_encoder)
        assert post_adjoined.shape == (batch_size, 1 + seq_length, self.n_mels)
        assert not torch.any(torch.isnan(post_adjoined))
        
        res = post_adjoined[:,1:,:]
        assert res.shape == (batch_size, seq_length, self.n_mels)
        assert not torch.any(torch.isnan(res))
        
        return res