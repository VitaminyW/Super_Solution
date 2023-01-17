import utility
import data
import model
import loss
from option_fine_tune import args
from checkpoint import Checkpoint
from trainer import Trainer_fine_tune

utility.set_seed(args.seed)
checkpoint = Checkpoint(args)

if checkpoint.ok:
    loader = data.Data_fine_tune(args)
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer_fine_tune(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()
    checkpoint.done()
