import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from scripts import control_rnn_shared_length as controller
from scripts.run_rnn_shared_length import epoch_batches, data_for_t, save, grad_stats

PROTOCOL = json.loads((Path(__file__).resolve().parents[1] / 'configs/rnn_shared_length_ordered16.json').read_text())

class QueueTests(unittest.TestCase):
    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS required')
    def test_mps_gradient_norm_cpu_reduction(self):
        param = torch.nn.Parameter(torch.zeros(2, device='mps'))
        param.grad = torch.tensor([3., 4.], device='mps')
        self.assertEqual(grad_stats([param]), (5., 4.))

    def test_allocation_and_exact_once_order(self):
        counts = {int(t): n for t,n in PROTOCOL['ordered_counts'].items()}
        self.assertEqual(sum(counts.values()),600000)
        self.assertTrue(all(n % 2 == 0 for n in counts.values()))
        for size in (128,256,512):
            for order in ('ascending','descending'):
                permutations,batches = epoch_batches(counts,size,order,torch.Generator().manual_seed(7))
                self.assertEqual([t for t,_ in batches],sorted([t for t,_ in batches],reverse=order=='descending'))
                for t,n in counts.items():
                    indices=torch.cat([permutations[t][start:start+size] for bt,start in batches if bt==t])
                    self.assertEqual(indices.sort().values.tolist(),list(range(n)))

    def test_data_balance_and_normalization(self):
        x,y=data_for_t(50,144,PROTOCOL)
        self.assertEqual(int((y==0).sum()),72)
        self.assertTrue(torch.isfinite(x).all())
        self.assertTrue(torch.allclose(x.mean(1),torch.zeros(144),atol=1e-6))
        self.assertTrue(torch.allclose(x.abs().max(1).values,torch.ones(144)))

    def test_queue_terminal_and_interruption_rules(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(controller,'BASE',Path(temp)), patch.object(controller,'training_processes',return_value=[]), patch.object(controller,'process_alive',return_value=False):
            base=Path(temp)
            protocol=dict(PROTOCOL,code_sha256={})
            save(base/'protocol.json',protocol)
            def fake_launch(candidate,folder):
                folder.mkdir()
                save(folder/'status.json',dict(status='starting',started_at=0))
                return dict(status='starting',new_process_started=True)
            with patch.object(controller,'launch',side_effect=fake_launch) as launch:
                for i,candidate in enumerate(protocol['candidates']):
                    result=controller.tick()
                    self.assertEqual(result['candidate']['id'],candidate['id'])
                    self.assertTrue(result['new_process_started'])
                    self.assertEqual(controller.tick()['status'],'needs_attention')
                    state=dict(status='completed',completed_epochs=200) if i%2==0 else dict(status='numerical_failure',T=50,epoch=1,batch=2)
                    save(base/candidate['id']/'status.json',state)
                result=controller.tick()
                self.assertEqual(result['status'],'all_done')
                self.assertTrue(result['pause_automation'])
                self.assertEqual(launch.call_count,16)
                self.assertEqual(len(json.loads((base/'summary.json').read_text())['runs']),16)
                self.assertEqual(controller.tick()['status'],'all_done')
                self.assertEqual(launch.call_count,16)

    def test_terminal_process_must_exit(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(controller,'BASE',Path(temp)), patch.object(controller,'process_alive',return_value=True), patch.object(controller,'launch') as launch:
            base=Path(temp)
            save(base/'protocol.json',dict(PROTOCOL,code_sha256={}))
            folder=base/PROTOCOL['candidates'][0]['id']; folder.mkdir()
            save(folder/'status.json',dict(status='completed',completed_epochs=200,pid=123))
            self.assertEqual(controller.tick()['status'],'running')
            launch.assert_not_called()

if __name__=='__main__':
    unittest.main()
