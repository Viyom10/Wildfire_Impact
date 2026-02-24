"""
MHEIDS Test Suite — Model & Loss Tests

Tests model forward passes, loss functions, and metrics.
NOTE: These tests use random weights (not pretrained) to verify shapes and
that no NaN / errors occur.  They do NOT test accuracy.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import numpy as np


# ── Model forward-pass tests ────────────────────────────────────────

def _random_pair(C=9, H=64, W=64, batch=1):
    """Return a random before/after tensor pair on CPU."""
    x1 = torch.randn(batch, C, H, W)
    x2 = torch.randn(batch, C, H, W)
    return x1, x2


class TestUNet:
    def test_forward_shape(self):
        from models.unet import Unet
        # UNet.forward does torch.cat((x1,x2),1) so first conv needs 2*C channels
        model = Unet(input_nbr=18, label_nbr=2)
        x1, x2 = _random_pair()
        out = model(x1, x2)
        assert out.shape == (1, 2, 64, 64)
        assert not torch.isnan(out).any()


class TestFCEFConc:
    def test_forward_shape(self):
        from models.fc_ef_conc import FC_EF_conc
        model = FC_EF_conc(input_nbr=9, label_nbr=2)
        x1, x2 = _random_pair()
        out = model(x1, x2)
        assert out.shape == (1, 2, 64, 64)


class TestFCEFDiff:
    def test_forward_shape(self):
        from models.fc_ef_diff import FC_EF_diff
        model = FC_EF_diff(input_nbr=9, label_nbr=2)
        x1, x2 = _random_pair()
        out = model(x1, x2)
        assert out.shape == (1, 2, 64, 64)


class TestSNUNet:
    def test_forward_shape(self):
        from models.snunet import SNUNet_ECAM
        model = SNUNet_ECAM(in_channels=9, out_ch=2, base_channel=32)
        x1, x2 = _random_pair()
        out = model(x1, x2)
        assert out.shape == (1, 2, 64, 64)


class TestADHR:
    def test_forward_shape(self):
        from models.adhr_cdnet import ADHR
        model = ADHR(in_channels=9, num_classes=2)
        x1, x2 = _random_pair()
        out = model(x1, x2)
        assert out.shape == (1, 2, 64, 64)


# ── Loss function tests ─────────────────────────────────────────────

class TestDiceLoss:
    def test_perfect_prediction(self):
        from losses.dice import DiceLoss
        loss_fn = DiceLoss(ignore_index=2, use_softmax=True)
        # create perfect one-hot prediction
        label = torch.zeros(1, 64, 64, dtype=torch.long)
        label[0, 16:48, 16:48] = 1
        pred = torch.zeros(1, 2, 64, 64)
        pred[0, 0] = (label == 0).float() * 10   # high logit for class 0
        pred[0, 1] = (label == 1).float() * 10   # high logit for class 1
        loss = loss_fn(pred, label)
        assert loss.item() < 0.05

    def test_worst_prediction(self):
        from losses.dice import DiceLoss
        loss_fn = DiceLoss(ignore_index=2, use_softmax=True)
        label = torch.ones(1, 64, 64, dtype=torch.long)
        pred = torch.zeros(1, 2, 64, 64)
        pred[0, 0] = 10.0   # predict all class 0, but label is all class 1
        loss = loss_fn(pred, label)
        assert loss.item() > 0.8

    def test_ignore_index(self):
        from losses.dice import DiceLoss
        loss_fn = DiceLoss(ignore_index=2, use_softmax=True)
        label = torch.full((1, 64, 64), 2, dtype=torch.long)  # all ignored
        pred = torch.randn(1, 2, 64, 64)
        loss = loss_fn(pred, label)
        assert torch.isfinite(loss)


class TestBCEandDiceLoss:
    def test_combined_loss(self):
        from losses.bce_and_dice import BCEandDiceLoss
        loss_fn = BCEandDiceLoss(weights=torch.tensor([1.0, 1.0]),
                                 ignore_index=2, use_softmax=True)
        label = torch.zeros(1, 64, 64, dtype=torch.long)
        pred = torch.randn(1, 2, 64, 64)
        loss = loss_fn(pred, label)
        assert torch.isfinite(loss) and loss.item() > 0


# ── Metrics tests ────────────────────────────────────────────────────

class TestConfusionMatrix:
    def test_perfect_accuracy(self):
        from utils import MyConfusionMatrix
        cm = MyConfusionMatrix(num_classes=3, ignore_index=2, device='cpu')
        preds  = torch.tensor([0, 0, 1, 1, 0])
        labels = torch.tensor([0, 0, 1, 1, 0])
        cm.compute(preds, labels)
        acc = cm.accuracy()
        assert torch.all(acc >= 0.99)

    def test_f1_known(self):
        from utils import MyConfusionMatrix
        cm = MyConfusionMatrix(num_classes=3, ignore_index=2, device='cpu')
        # TP=2, FP=1, FN=1 for class 1
        preds  = torch.tensor([0, 1, 1, 1, 0])
        labels = torch.tensor([0, 1, 0, 1, 1])
        cm.compute(preds, labels)
        f1 = cm.f1_score()
        # F1 for class 1 = 2*2/(2*2+1+1) = 4/6 ≈ 0.667
        assert f1[1].item() == pytest.approx(0.667, abs=0.01)

    def test_ignore_index_excluded(self):
        from utils import MyConfusionMatrix
        cm = MyConfusionMatrix(num_classes=3, ignore_index=2, device='cpu')
        preds  = torch.tensor([0, 2, 1])
        labels = torch.tensor([0, 2, 1])
        matrix = cm.compute(preds, labels)
        # After dropping ignore_index, matrix is 2×2
        assert matrix.shape == (2, 2)
