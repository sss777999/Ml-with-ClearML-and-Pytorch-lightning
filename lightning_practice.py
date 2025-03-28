import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse

# Класс датасета
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.FloatTensor(data.values)
        self.targets = torch.LongTensor(targets.values)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Класс DataModule
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        
    def prepare_data(self):
        # Загрузка данных как в ноутбуке
        # Предполагая, что данные находятся в том же каталоге
        pass
    
    def setup(self, stage=None):
        # Загрузка данных
        train = pd.read_csv('data/sign_mnist_train.csv')
        
        # Разделение признаков и целевой переменной
        X = train.drop('label', axis=1)
        y = train['label']
        
        # Разделение на обучающую и валидационную выборки
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Создание датасетов
        self.train_dataset = CustomDataset(self.X_train, self.y_train)
        self.val_dataset = CustomDataset(self.X_val, self.y_val)
        
        # Сохраняем тестовый датасет для инференса
        self.test = pd.read_csv('data/sign_mnist_test.csv')
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

# Класс модели
class SimpleModel(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_size, 128)
        self.layer2 = torch.nn.Linear(128, 64)
        self.layer3 = torch.nn.Linear(64, 25)  # У нас 25 классов для sign language
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        return self.layer3(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        
        # Логируем метрики
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        
        # Логируем метрики
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast_dev_run', type=lambda x: (str(x).lower() == 'true'), default=False)
    args = parser.parse_args()

    # Создание DataModule
    dm = CustomDataModule(batch_size=64)
    dm.setup()
    
    # Получаем размерность входных данных из обучающего датасета
    input_size = dm.X_train.shape[1]
    
    # Создание модели
    model = SimpleModel(input_size=input_size)
    
    # Настройка чекпоинтов
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    
    # Создание тренера
    trainer = pl.Trainer(
        max_epochs=10,
        fast_dev_run=args.fast_dev_run,
        callbacks=[checkpoint_callback]
    )
    
    try:
        # Запуск обучения
        trainer.fit(model, dm)
        
        if args.fast_dev_run:
            print("Тестовый прогон успешно пройден")
    except Exception as e:
        if args.fast_dev_run:
            print("Тестовый прогон завершился с ошибкой")
            return
        else:
            raise e
    
    # Если не fast_dev_run или тестовый прогон прошел успешно
    if not args.fast_dev_run:
        # Загрузка лучшей модели
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            best_model = SimpleModel.load_from_checkpoint(best_model_path, input_size=input_size)
            print(f"Загружена лучшая модель из {best_model_path}")
            
            # Инференс на одном образце из теста
            test_features = dm.test.drop('label', axis=1)
            test_sample = torch.FloatTensor(test_features.iloc[0].values.reshape(1, -1))
            # Перемещаем тензор на тот же девайс, что и модель
            device = best_model.device
            test_sample = test_sample.to(device)
            
            with torch.no_grad():
                prediction = best_model(test_sample)
                predicted_class = torch.argmax(prediction, dim=1).item()
                probabilities = torch.nn.functional.softmax(prediction, dim=1)
                print(f"Предсказание для первого тестового образца: класс {predicted_class}")
                print(f"Вероятности: {probabilities.cpu().numpy()}")

if __name__ == "__main__":
    main() 