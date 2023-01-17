import utility
import data
import model
import loss
from option_reconstrution import args
from checkpoint import Checkpoint
from trainer import Trainer_reconstruction

utility.set_seed(args.seed)
checkpoint = Checkpoint(args)

if checkpoint.ok:
    loader = data.Data_reconstruction(args)
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer_reconstruction(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.reconstruction()
    checkpoint.done()