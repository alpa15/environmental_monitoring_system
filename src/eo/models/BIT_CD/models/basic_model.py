import os
from pathlib import Path

import torch

from misc.imutils import save_image
from models.networks import *


class CDEvaluator():

    def __init__(self, args):

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0]
                                   if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")

        print(self.device)

        self.checkpoint_dir = args.checkpoint_dir

        self.pred_dir = args.output_folder
        os.makedirs(self.pred_dir, exist_ok=True)

    def load_checkpoint(self, checkpoint_name="best_ckpt.pt"):
        ckpt_path = Path(__file__).resolve().parent.parent / self.checkpoint_dir / checkpoint_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"no such checkpoint {checkpoint_name}")
        
        checkpoint = torch.load(str(ckpt_path), map_location=self.device)

        if not isinstance(checkpoint, dict) or "model_G_state_dict" not in checkpoint:
            raise ValueError("Checkpoint format not recognized")

        # carica solo ciò che ti serve
        state = checkpoint["model_G_state_dict"]
        self.net_G.load_state_dict(state, strict=True)
        self.net_G.to(self.device)

        self.best_val_acc = checkpoint.get("best_val_acc", None)
        self.best_epoch_id = checkpoint.get("best_epoch_id", None)

        return self.net_G


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.shape_h = img_in1.shape[-2]
        self.shape_w = img_in1.shape[-1]
        self.G_pred = self.net_G(img_in1, img_in2)
        return self._visualize_pred()

    def eval(self):
        self.net_G.eval()

    def _save_predictions(self):
        """
        保存模型输出结果，二分类图像
        """

        preds = self._visualize_pred()
        name = self.batch['name']
        for i, pred in enumerate(preds):
            file_name = os.path.join(
                self.pred_dir, name[i].replace('.jpg', '.png'))
            pred = pred[0].cpu().numpy()
            save_image(pred, file_name)

