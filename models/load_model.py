from models.HTGN import HTGN


def load_model(args):
    if args.model == 'HTGN':
        model = HTGN(args)
    else:
        raise Exception('pls define the model')
    return model
