from email_tools import Email

from data.inp.decorators import error_notif
from nebfir.config.options import get_options
from nebfir.trainers.trainer_manager import get_trainer


def main():
    args = get_options()

    @error_notif(yes=args.error_notif)
    def main_():
        trainer_class = get_trainer('base_dataloader')
        
        trainer = trainer_class(args=args)

        if args.dry:
            trainer.train(dry_len=1)
        if args.train:
            trainer.train()
        if args.test:
            trainer.test(batch_size=args.batch_size, progress=args.model_weights)

    main_()
    
    if args.error_notif: Email().send('Finnished')
        
    
if __name__ == "__main__":
    main()
    
   