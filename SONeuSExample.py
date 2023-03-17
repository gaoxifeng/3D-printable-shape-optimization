from ShapeOptNeuS import ShapeOptNeuS
import torch, logging
import argparse
from Render_Fields_modified import Render_Fields
# torch.set_default_dtype(torch.float64)
def Toy_Example(res=(180,60,4), volfrac=0.3):
    nelx, nely, nelz = res
    phiTensor = -torch.ones((nelx, nely, nelz)).cuda()
    phiFixedTensor = torch.ones((nelx + 1, nely + 1, nelz + 1)).cuda()
    phiFixedTensor[0,:,:] = -1
    f = torch.zeros((3, nelx + 1, nely + 1, nelz + 1)).cuda()
    f[1, -1, 0, :] = -1
    lam = 0.3 / 0.52
    mu = 1 / 2.6
    return res, volfrac, (phiTensor, phiFixedTensor, f, lam, mu)
if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/thin_structure.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='bmvs_bear')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    args.conf = '/home/yteng/Dropbox/Research/TencentNA/Test/confs/wmask_f16_modified.conf'
    args.field_shape = [64, 64, 64]
    args.method = 'Network'
    args.mode = 'train'
    args.case = 'cow'
    args.mesh_dir = '/home/yteng/Dropbox/Research/TencentNA/Test/data/f16/f16.obj'
    args.out_dir = 'asd'
    # args.MTL = './data/bunny/dancer_diffuse.mtl'
    args.MTL = None
    Field_runner = Render_Fields(args.mesh_dir, args.out_dir, args.conf, args.field_shape, args.MTL, args.method, args.mode, args.case)

    PSO = ShapeOptNeuS(Field_runner, 0.5)
    _, _, params = Toy_Example()
    PSO.run(*params)