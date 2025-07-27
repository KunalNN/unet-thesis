# src/engine/train.py
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from src.models.unet import build_unet_sr
from src.datasets.div2k import get_div2k_dataset
from src.utils.loss import charbonnier_loss

def main(args):
    # 1) build data
    train_ds = get_div2k_dataset(args.data_root, scale=args.scale,
                                 hr_crop=args.crop, batch_size=args.batch_size, training=True)
    val_ds   = get_div2k_dataset(args.data_root, scale=args.scale,
                                 hr_crop=args.crop, batch_size=args.batch_size, training=False)

    # 2) build model
    model = build_unet_sr(input_shape=(None,None,3),
                         scale=args.scale,
                         base_filters=args.base_filters)
    model.compile(
        optimizer="adam",
        loss=charbonnier_loss(),
        metrics=[]
    )

    # 3) callbacks
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    chk_cb = ModelCheckpoint(
        filepath=os.path.join(args.checkpoint_dir, "best_unet.h5"),
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )
    tb_cb = TensorBoard(log_dir=args.log_dir)

    # 4) train
    model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        verbose=1,
        callbacks=[chk_cb, tb_cb]
    )

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",   type=str, required=True)
    p.add_argument("--scale",       type=int, default=4)
    p.add_argument("--crop",        type=int, default=256)
    p.add_argument("--batch-size",  type=int, default=16)
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--base-filters",type=int, default=64)
    p.add_argument("--checkpoint-dir",type=str, default="outputs/checkpoints")
    p.add_argument("--log-dir",     type=str, default="outputs/logs")
    args = p.parse_args()
    main(args)
