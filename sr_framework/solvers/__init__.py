
def create_solver(opt):
    if opt['mode'] == 'SR':
        from .SRSolver import SRSolver as solver
    else:
        raise NotImplementedError('unrecognized mode: {}'.format(opt['mode']))
    
    return solver(opt)
