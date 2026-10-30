"""Tiny optimizer tests; no experiment data, training runs, or W&B."""
import io
import unittest

import torch

from src.grid50_safe_adam import Grid50SafeAdam


class SafeAdamTests(unittest.TestCase):
    def compare_adam(self, device, **kwargs):
        p = torch.nn.Parameter(torch.tensor([.3, -.8, 1.2], device=device))
        q = torch.nn.Parameter(p.detach().clone())
        reference = torch.optim.Adam([p], foreach=False, fused=False, **kwargs)
        safe = Grid50SafeAdam([q], **kwargs)
        for i in range(10):
            p.grad = torch.tensor([.1 + i * .02, -.4, .7], device=device)
            q.grad = p.grad.clone()
            reference.step()
            safe.step()
            self.assertTrue(torch.isfinite(q).all())
            torch.testing.assert_close(p, q, rtol=2e-6, atol=2e-7)
            self.assertEqual(safe.state[q]['step'].device.type, 'cpu')
            for key in ('exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(reference.state[p][key], safe.state[q][key],
                                           rtol=2e-6, atol=2e-7)

    def test_cpu_adam_parity(self):
        self.compare_adam('cpu')
        self.compare_adam('cpu', lr=.003, betas=(.8, .97), eps=1e-5)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps_adam_parity(self):
        self.compare_adam('mps')

    def roundtrip(self, device):
        p = torch.nn.Parameter(torch.tensor([.3, -.8], device=device))
        original = Grid50SafeAdam([p], lr=.002, betas=(.8, .99), eps=1e-6)
        p.grad = torch.tensor([.1, -.3], device=device)
        original.step()
        # Exact legacy optimizer state format, serialized as in checkpoints.
        self.assertEqual(set(original.state[p]), {'step', 'exp_avg', 'exp_avg_sq'})
        self.assertEqual(set(original.param_groups[0]), {'params', 'lr', 'betas', 'eps'})
        buffer = io.BytesIO()
        torch.save(original.state_dict(), buffer)
        buffer.seek(0)
        restored_state = torch.load(buffer, map_location='cpu', weights_only=True)
        q = torch.nn.Parameter(p.detach().clone())
        restored = Grid50SafeAdam([q])
        restored.load_state_dict(restored_state)
        self.assertEqual(restored.state[q]['step'].device.type, 'cpu')
        for i in range(3):
            p.grad = torch.tensor([.2 + i, -.7], device=device)
            q.grad = p.grad.clone()
            original.step()
            restored.step()
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(original.state[p][key], restored.state[q][key],
                                           rtol=0, atol=0)

    def test_cpu_checkpoint_roundtrip(self):
        self.roundtrip('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps_checkpoint_roundtrip(self):
        self.roundtrip('mps')

    def test_large_finite_gradient(self):
        # g**2 exceeds float32; (1-beta2)*g**2 is still representable.
        expected = .001 * torch.tensor([7.24e19], dtype=torch.float64).square()
        devices = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                p = torch.nn.Parameter(torch.ones(1, device=device))
                opt = Grid50SafeAdam([p])
                p.grad = torch.full_like(p, 7.24e19)
                opt.step()
                self.assertTrue(torch.isfinite(p).all())
                self.assertTrue(torch.isfinite(opt.state[p]['exp_avg_sq']).all())
                torch.testing.assert_close(opt.state[p]['exp_avg_sq'].cpu().double(),
                                           expected, rtol=3e-6, atol=0)

    def test_closure_and_missing_gradient(self):
        p = torch.nn.Parameter(torch.tensor([1.]))
        unused = torch.nn.Parameter(torch.tensor([2.]))
        opt = Grid50SafeAdam([p, unused])
        def closure():
            opt.zero_grad()
            loss = p.square().sum()
            loss.backward()
            return loss
        self.assertEqual(opt.step(closure).item(), 1.)
        self.assertNotIn(unused, opt.state)
        self.assertEqual(unused.item(), 2.)

    def test_unsupported_options_remain_unsupported(self):
        p = torch.nn.Parameter(torch.ones(1))
        for option in ({'weight_decay': .01}, {'amsgrad': True}):
            with self.assertRaises(TypeError):
                Grid50SafeAdam([p], **option)
        for option in ({'lr': -1}, {'eps': -1}, {'betas': (.9, 1.)}):
            with self.assertRaises(ValueError):
                Grid50SafeAdam([p], **option)


if __name__ == '__main__':
    unittest.main()
