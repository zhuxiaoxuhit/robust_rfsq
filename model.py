from torchvision import models
import torch
import torch.nn as nn
from quantizers import VectorQuantizeEMA, FSQ, LFQ, RFSQ

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # 根据量化器类型确定输出维度
        output_dim = len(args.levels) if args.quantizer in ['fsq', 'rfsq'] else args.embed_dim
        
        # 统一使用简单架构确保公平对比
        self.blocks = nn.Sequential(
            nn.Conv2d(3, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, output_dim, 1),
        )
        
    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # 根据量化器类型确定输入维度
        input_dim = len(args.levels) if args.quantizer in ['fsq', 'rfsq'] else args.embed_dim
        
        # 统一使用简单架构确保公平对比
        self.blocks = nn.Sequential(
            nn.Conv2d(input_dim, 256, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
        )

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        if args.quantizer == 'ema' or args.quantizer == 'origin':
            self.quantize_t = VectorQuantizeEMA(args, args.embed_dim, args.n_embed)

        elif args.quantizer == 'lfq':
            self.quantize_t = LFQ(codebook_size = 2**args.lfq_dim, dim = args.lfq_dim, entropy_loss_weight=args.entropy_loss_weight, commitment_loss_weight=args.codebook_loss_weight)
            # args.embed_dim = args.lfq_dim
        elif args.quantizer == 'fsq':
            self.quantize_t = FSQ(levels=args.levels)
            # args.embed_dim = len(args.levels)
        elif args.quantizer == 'rfsq':
            self.quantize_t = RFSQ(
                num_stages=args.rfsq_stages,
                strategy=args.rfsq_strategy,
                fsq_levels=args.levels,
                dim=len(args.levels) if args.rfsq_strategy == 'layernorm' else None,
                initial_scale=args.rfsq_initial_scale
            )
        else:
            print('quantizer error!')
            exit()

        self.enc = Encoder(args)
        self.dec = Decoder(args)
        
    def forward(self, input, return_id=True):
        quant_t, diff, id_t = self.encode(input)
        dec = self.dec(quant_t)
        if return_id:
            return dec, diff, id_t
        return dec, diff

    def encode(self, input):
        logits = self.enc(input)
        if self.args.quantizer == 'ema'  or self.args.quantizer == 'origin':
            quant_t, diff_t, id_t = self.quantize_t(logits)
            # quant_t = quant_t.permute(0, 3, 1, 2) have change the dimension in quantizer
            diff_t = diff_t.unsqueeze(0)
        
        elif self.args.quantizer == 'fsq':
            quant_t, id_t = self.quantize_t(logits)
            diff_t = torch.tensor(0.0).cuda().float()
        
        elif self.args.quantizer == 'rfsq':
            quant_t, id_t = self.quantize_t(logits)
            diff_t = torch.tensor(0.0).cuda().float()
        
        elif self.args.quantizer == 'lfq':
            # quantized, indices, entropy_aux_loss = quantizer(image_feats)
            quant_t, id_t, diff_t = self.quantize_t(logits)
        return quant_t, diff_t, id_t

    def decode(self, code):
        return self.dec(code)

    def decode_code(self, code_t):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        dec = self.dec(quant_t)
        return dec
