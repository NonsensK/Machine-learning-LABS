"""Run python -m unittest discover -s tests."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from models import CNN, apply_masks, compact_filters, distillation_loss, prune_weights


class CompressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_exact_global_count_and_fixed_mask(self):
        model,masks = prune_weights(CNN((2,4),8),0.75,"global")
        count = sum(mask.numel() for mask in masks.values())
        self.assertEqual(sum((mask==0).sum().item() for mask in masks.values()),int(.75*count))
        optimizer = torch.optim.Adam(model.parameters())
        model(torch.randn(3,1,28,28)).sum().backward()
        optimizer.step()
        apply_masks(model,masks)
        for name,param in model.named_parameters():
            if name in masks:
                self.assertEqual(torch.count_nonzero(param[masks[name]==0]).item(),0)

    def test_layer_targets(self):
        model,masks = prune_weights(CNN((2,4),8),.5,"layer")
        for mask in masks.values():
            self.assertEqual((mask==0).sum().item(),int(.5*mask.numel()))

    def test_compact_matches_removed_filters(self):
        original = CNN((4,8),12)
        compact = compact_filters(original,.5)
        keep1 = torch.argsort(original.conv1.weight.flatten(1).norm(dim=1),descending=True)[:2].sort().values
        keep2 = torch.argsort(original.conv2.weight.flatten(1).norm(dim=1),descending=True)[:4].sort().values
        x = torch.randn(3,1,28,28)
        first = torch.nn.functional.max_pool2d(original.conv1(x).relu(),2)
        mask1 = torch.zeros(4); mask1[keep1] = 1
        first *= mask1[None,:,None,None]
        second = torch.nn.functional.max_pool2d(original.conv2(first).relu(),2)
        mask2 = torch.zeros(8); mask2[keep2] = 1
        second *= mask2[None,:,None,None]
        reference = original.fc2(original.fc1(second.flatten(1)).relu())
        torch.testing.assert_close(compact(x),reference)
        self.assertLess(sum(p.numel() for p in compact.parameters()),sum(p.numel() for p in original.parameters()))

    def test_kl_direction_and_teacher_detached(self):
        student = torch.tensor([[1.,2.,-1.]],requires_grad=True)
        teacher = torch.tensor([[3.,0.,1.]],requires_grad=True)
        labels = torch.tensor([1])
        loss = distillation_loss(student,teacher,labels,.5,4)
        p = torch.softmax(student/4,dim=1)
        q = torch.softmax(teacher.detach()/4,dim=1)
        expected = .5*torch.nn.functional.cross_entropy(student,labels)+8*(p*(p.log()-q.log())).sum()
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)


if __name__ == "__main__":
    unittest.main()
