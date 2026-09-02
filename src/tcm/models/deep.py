"""Derin öğrenme modeli: 1B-CNN + GRU (Faz 05).

Gradyan artırmadan farkı: biz özniteliği tanımlamıyoruz. Orada "RMS hesapla,
3. mertebe enerjisini al" diyorduk; burada ağ ham sinyalden ne bakacağını
kendisi öğreniyor.

Mimari
------
    ham sinyal (6 kanal x 9000 örnek)
      -> Conv1D bloklar   : yerel örüntüleri bulur, diziyi kısaltır
      -> GRU              : zaman içindeki değişimi takip eder
      -> kesme parametreleri eklenir (evrişimden geçmez - sinyal değiller)
      -> yoğun katman     -> VB tahmini

Beklenti
--------
DÜŞÜK. NASA'da 145 örnek var; derin ağlar tipik olarak binlerce örnekle
çalışır. Ezberleme riski yüksek. Bu yüzden model kasıtlı olarak küçük
tutuldu ve düzenlileştirme (dropout, weight decay) yüksek.

"Denedik, gradyan artırmayı geçemedi" de geçerli bir bulgudur ve nedeni
veri ölçeğiyle açıklanabilir.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch kurulu değilse
    TORCH_AVAILABLE = False
    torch = None
    nn = object


def require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch kurulu değil. Kurmak için:\n"
            "    pip install -r requirements-dl.txt"
        )


class CNNGRUWearModel(nn.Module if TORCH_AVAILABLE else object):
    """Sinyalden VB tahmini yapan küçük evrişimli-tekrarlayan ağ."""

    def __init__(
        self,
        n_channels: int = 6,
        n_parameters: int = 5,
        conv_channels: tuple[int, ...] = (16, 32, 32),
        gru_hidden: int = 32,
        dropout: float = 0.3,
    ) -> None:
        require_torch()
        super().__init__()

        blocks = []
        in_channels = n_channels
        for out_channels in conv_channels:
            blocks += [
                nn.Conv1d(in_channels, out_channels, kernel_size=7,
                          stride=1, padding=3),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                # Havuzlama diziyi 4 kat kısaltır: 9000 -> 2250 -> 562 -> 140
                nn.MaxPool1d(4),
                nn.Dropout(dropout),
            ]
            in_channels = out_channels
        self.conv = nn.Sequential(*blocks)

        self.gru = nn.GRU(
            input_size=conv_channels[-1],
            hidden_size=gru_hidden,
            batch_first=True,
        )

        # Kesme parametreleri evrişime girmez; GRU çıktısına eklenir.
        self.head = nn.Sequential(
            nn.Linear(gru_hidden + n_parameters, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, signal, parameters):
        # signal: (yığın, kanal, uzunluk)
        features = self.conv(signal)
        # GRU (yığın, uzunluk, kanal) bekler
        sequence = features.transpose(1, 2)
        _, hidden = self.gru(sequence)
        combined = torch.cat([hidden[-1], parameters], dim=1)
        return self.head(combined).squeeze(1)


class SignalStandardiser:
    """Kanal bazında ortalama/standart sapma normalizasyonu.

    İstatistikler YALNIZCA eğitim kümesinden hesaplanır; test kümesinden
    hesaplamak sızıntı olur.
    """

    def __init__(self) -> None:
        self.mean_ = None
        self.std_ = None

    def fit(self, signals: np.ndarray) -> "SignalStandardiser":
        # signals: (örnek, kanal, uzunluk)
        self.mean_ = signals.mean(axis=(0, 2), keepdims=True)
        self.std_ = signals.std(axis=(0, 2), keepdims=True)
        self.std_[self.std_ < 1e-9] = 1.0
        return self

    def transform(self, signals: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Önce fit() çağrılmalı")
        return (signals - self.mean_) / self.std_

    def fit_transform(self, signals: np.ndarray) -> np.ndarray:
        return self.fit(signals).transform(signals)


def train_model(
    model,
    signals: np.ndarray,
    parameters: np.ndarray,
    targets: np.ndarray,
    epochs: int = 120,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    seed: int = 42,
    verbose: bool = False,
    validation: tuple | None = None,
    patience: int = 20,
):
    """Modeli eğitir.

    ``validation`` verilmezse sabit ``epochs`` kadar eğitilir. Çapraz
    doğrulama kurgusunda durum budur: her katlamada ayıracak kadar veri yok,
    onun yerine kapasite küçük tutulur ve düzenlileştirme yükseltilir.

    ``validation`` verilirse (``(sinyal, parametre, hedef)`` üçlüsü) her
    epoch sonunda doğrulama MAE'si ölçülür, en iyi ağırlıklar saklanır ve
    ``patience`` epoch boyunca iyileşme olmazsa eğitim durdurulup **en iyi
    ağırlıklara geri dönülür**. Son epoch'un ağırlıkları değil, en iyisi
    döndürülür - aksi halde erken durdurmanın anlamı kalmaz.
    """
    require_torch()
    torch.manual_seed(seed)

    device = torch.device("cpu")
    model = model.to(device)

    x_signal = torch.tensor(signals, dtype=torch.float32, device=device)
    x_param = torch.tensor(parameters, dtype=torch.float32, device=device)
    y = torch.tensor(targets, dtype=torch.float32, device=device)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.SmoothL1Loss()  # aykırı değerlere MSE'den dayanıklı

    n = len(y)
    generator = torch.Generator().manual_seed(seed)

    validation_tensors = None
    if validation is not None:
        v_signal, v_param, v_target = validation
        validation_tensors = (
            torch.tensor(v_signal, dtype=torch.float32, device=device),
            torch.tensor(v_param, dtype=torch.float32, device=device),
            torch.tensor(v_target, dtype=torch.float32, device=device),
        )

    best_loss, best_state, best_epoch, waited = float("inf"), None, 0, 0

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n, generator=generator)
        total = 0.0
        for start in range(0, n, batch_size):
            index = order[start:start + batch_size]
            if len(index) < 2:  # BatchNorm tek örnekle çalışmaz
                continue
            optimiser.zero_grad()
            prediction = model(x_signal[index], x_param[index])
            loss = loss_fn(prediction, y[index])
            loss.backward()
            optimiser.step()
            total += loss.detach().item() * len(index)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch + 1:3d}  kayıp {total / n:8.2f}")

        if validation_tensors is None:
            continue

        model.eval()
        with torch.no_grad():
            v_signal, v_param, v_target = validation_tensors
            v_mae = float(torch.mean(torch.abs(
                model(v_signal, v_param) - v_target)).item())

        if v_mae < best_loss - 1e-9:
            best_loss, best_epoch, waited = v_mae, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.best_epoch_ = best_epoch
        model.best_validation_mae_ = best_loss

    return model


def predict(model, signals: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    require_torch()
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.tensor(signals, dtype=torch.float32),
            torch.tensor(parameters, dtype=torch.float32),
        )
    return prediction.numpy()
