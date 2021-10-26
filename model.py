
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader



class TestModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.build()
        self.batch_size = 4

    @classmethod
    def config_path(cls):
        return "configs/models/cvlg/defaults.yaml"

    def build(self):
        config_encoder = BertConfig()
        config_decoder = BertConfig()
        config_encoder.max_position_embeddings = 500
        config_decoder.max_position_embeddings = 100
        self.codec_config = EncoderDecoderConfig.from_encoder_decoder_configs(
        config_encoder, config_decoder
        )
        self.encoderdecoder = EncoderDecoderModel(config=self.codec_config)
        self.criterion = ImageCrossEntropyLoss()
        self.metric = None


    def configure_optimizers(self):
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        optimizer = DeepSpeedCPUAdam([p for p in self.parameters() if p.requires_grad],
                                        lr=5e-4)
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=100,
                                                    num_training_steps=5000
                                                    )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


    def forward(self, sample_list):
        return None

    def training_step(self, sample_list, batch_idx):
    
        decoder_input_ids = sample_list["input_ids"].new_full([sample_list["input_ids"].shape[0], 100], 20)
        caption_ids = sample_list["input_ids"]
        inputs_embeds = self.encoderdecoder.encoder.embeddings.word_embeddings(
            caption_ids)
        
        decoder_output = self.encoderdecoder(
            decoder_input_ids=decoder_input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
        )
        
        logits = decoder_output["logits"]
        model_output = {}
        model_output["scores"] = logits
        sample_list["targets"] = decoder_input_ids[:,:-1]

        loss = self.criterion(sample_list, model_output)
        self.log("cross_entropy", loss)

        return loss

    def validation_step(self, sample_list, batch_idx):
        return None

        
class ImageCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the cross entropy loss for captions.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.
        """
        scores = model_output["scores"]
        targets = sample_list["targets"]

        scores[scores != scores] = 0.0
        scores[scores == float("inf")] = torch.finfo(scores.dtype).max
        
        decode_lengths = [targets.size(1)] * targets.size(0)
        if torch.__version__ >= "1.1":
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=False).data
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True, enforce_sorted=False
            ).data
        else:
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        loss = F.cross_entropy(scores, targets)

        return loss
        
        
class TestDataset(Dataset):
    def __init__(self, dataset_type, local=False) -> None:
        super().__init__()

    def __getitem__(self, idx):
        current_sample = {}
        current_sample["input_ids"] = torch.arange(0,500)
        return current_sample

    def __len__(self):
        return 5



class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int=32, num_workers: int=2):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_class = TestDataset
        self.cvlg_train = self.dataset_class(dataset_type="train")
        self.cvlg_val = self.dataset_class(dataset_type="val")
        
    def train_dataloader(self):
        return DataLoader(self.cvlg_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.cvlg_val, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.cvlg_val, batch_size=self.batch_size, num_workers=self.num_workers)
        



if __name__ == "__main__":
    from pytorch_lightning.plugins import DeepSpeedPlugin
    from pytorch_lightning.callbacks import ModelCheckpoint
    model = TestModel()
    checkpoint_callback = ModelCheckpoint(
        monitor="cross_entropy",
        filename="ckpt-{epoch:02d}-{cross_entropy:.2f}",
        save_top_k=3,
        mode="min",
    )
    dm = DataModule(batch_size=4)
    trainer = pl.Trainer(
        gpus=4, # ! Note that if change gpus to 1 the bug just disappeared
        num_nodes=1,
        accelerator="ddp",  
        default_root_dir=".",
        plugins=DeepSpeedPlugin(stage=3, cpu_offload=True, cpu_offload_params=True),
        precision=16,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, datamodule=dm)

