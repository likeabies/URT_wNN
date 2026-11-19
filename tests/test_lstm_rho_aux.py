import io
import unittest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from src.model import Paper2019LSTMClassifier
from src.lstm_rho_aux import Paper2019LSTMRhoAux
from src.train import ann_weighted_cross_entropy
from src.utils import set_seed


class RhoAuxTests(unittest.TestCase):
    def test_initial_parameters_and_shuffle_rng(self):
        set_seed(7);base=Paper2019LSTMClassifier();rng=torch.get_rng_state().clone()
        ds=TensorDataset(torch.arange(41));order=torch.cat([x[0] for x in DataLoader(ds,batch_size=8,shuffle=True)])
        set_seed(7);aux=Paper2019LSTMRhoAux()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in base.state_dict().items():
            self.assertTrue(torch.equal(value,aux.state_dict()[key]),key)
        aux_order=torch.cat([x[0] for x in DataLoader(ds,batch_size=8,shuffle=True)])
        self.assertTrue(torch.equal(order,aux_order))

    def smoke(self,device):
        set_seed(7);model=Paper2019LSTMRhoAux().to(device)
        x=torch.randn(8,250,device=device);y=torch.arange(8,device=device)%2
        rho=torch.tensor([1.,.99,.95,.9,.5,.2,1.,.99],device=device)
        logits,hat,h=model(x)
        self.assertEqual(logits.shape,(8,2));self.assertEqual(hat.shape,(8,));self.assertEqual(h.shape,(8,30))
        loss=ann_weighted_cross_entropy(logits,y,w1=1.,w2=.74375)+.1*F.mse_loss(hat,rho)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        for module in (model.lstm,model.fc,model.rho_head):
            for p in module.parameters():self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all())
        opt=torch.optim.Adam(model.parameters(),lr=.001);opt.step()
        with torch.no_grad():expected=model(x)
        stream=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),stream);stream.seek(0)
        cp=torch.load(stream,map_location=device,weights_only=True)
        reload=Paper2019LSTMRhoAux().to(device);reload.load_state_dict(cp['model'])
        with torch.no_grad():actual=reload(x)
        for a,b in zip(expected,actual):torch.testing.assert_close(a,b)
        with torch.no_grad():
            reload.rho_head.weight.zero_();reload.rho_head.bias.fill_(2.)
            self.assertTrue(torch.equal(reload(x)[1],torch.full((8,),2.,device=device)))

    def test_cpu_shapes_gradients_reload_raw_output(self):self.smoke('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_mps_shapes_gradients_reload(self):self.smoke('mps')


if __name__=='__main__':unittest.main()
