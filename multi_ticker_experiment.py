# ===============================================================
#  multi_ticker_experiment.py — Массовое тестирование по тикерам
# ---------------------------------------------------------------
#  Запускает llm_finance_predictor.py для нескольких инструментов:
#   • Перебирает все файлы с котировками в /Data
#   • Сохраняет результаты по каждому тикеру в /results
#   • Вычисляет усреднённые метрики по рынку
#
#  Удобно для оценки эффективности модели на множестве бумаг.
#
#  Автор: Михаил Шардин
#  Онлайн-визитка: https://shardin.name/?utm_source=python
# 
#  Репозиторий: https://github.com/empenoso/llm-stock-market-predictor
# ===============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import shutil

from llm_finance_predictor import LLMFinancialPredictor, WalkForwardValidator

class MultiTickerExperiment:
    """Проводит эксперименты на множестве тикеров"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config['data_dir'])
        self.results_dir = Path(config['results_dir'])
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        self.predictor = LLMFinancialPredictor(model_name=config['model_name'])
        self.validator = WalkForwardValidator(
            train_size=config['train_size'], 
            test_size=config['test_size'], 
            step_size=config['step_size']
        )
    
    def load_all_data(self) -> pd.DataFrame:
        """
        Загружает все файлы и объединяет в один DataFrame для пакетной обработки.
        """
        ticker_files = sorted(self.data_dir.glob("*.txt"))
        if self.config['max_tickers']:
            ticker_files = ticker_files[:self.config['max_tickers']]
        
        all_dfs = []
        print(f"Загрузка {len(ticker_files)} файлов...")
        for file in tqdm(ticker_files, desc="Чтение файлов"):
            try:
                df = self.predictor.load_data(file)
                df['ticker'] = file.stem.split('_')[0] 
                all_dfs.append(df)
            except Exception as e:
                print(f"Не удалось загрузить {file}: {e}")
        
        if not all_dfs:
            raise ValueError("Не найдено данных для обработки.")
            
        return pd.concat(all_dfs, ignore_index=True)

    def run(self):
        """
        Запускает полный цикл эксперимента: загрузка, пакетная обработка, обучение по тикерам, оценка.
        """
        print("="*80)
        print("НАЧАЛО МАСШТАБНОГО ЭКСПЕРИМЕНТА")
        print(f"Конфигурация: {self.config}")
        print("="*80)

        # 1. Пакетная загрузка
        try:
            combined_df = self.load_all_data()
        except ValueError as e:
            print(f"❌ Критическая ошибка: {e}")
            return

        # 2. Пакетная генерация признаков - ЕДИНСТВЕННЫЙ ВЫЗОВ НА ВСЕ ДАННЫЕ
        print("\n🔧 Пакетная генерация признаков для всех данных... Это может занять время.")
        features_df = self.predictor.feature_extractor.process_dataframe(combined_df)
        print(f"✅ Генерация признаков завершена. Всего строк с признаками: {len(features_df)}")
        
        # 3. Итерация по тикерам для обучения и валидации
        all_results = []
        tickers = features_df['ticker'].unique()
        
        for ticker_name in tqdm(tickers, desc="Обучение моделей по тикерам"):
            ticker_features = features_df[features_df['ticker'] == ticker_name].copy()
            
            if len(ticker_features) < (self.validator.train_size + self.validator.test_size):
                tqdm.write(f"⚠️ Пропуск {ticker_name}: недостаточно данных ({len(ticker_features)} строк)")
                all_results.append({'ticker': ticker_name, 'status': 'skipped_insufficient_data'})
                continue

            splits = self.validator.split(ticker_features)
            if not splits:
                all_results.append({'ticker': ticker_name, 'status': 'skipped_no_splits'})
                continue
            
            if self.config['max_folds']:
                splits = splits[:self.config['max_folds']]

            fold_results = []
            for i, (train_df, test_df) in enumerate(splits):
                train_texts, train_labels = self.predictor.prepare_dataset(train_df)
                test_texts, test_labels = self.predictor.prepare_dataset(test_df)

                if np.mean(train_labels) < 0.3 or np.mean(train_labels) > 0.7:
                    continue

                self.predictor.train(train_texts, train_labels, epochs=self.config['epochs'], batch_size=self.config['batch_size'])
                metrics = self.predictor.evaluate(test_texts, test_labels)
                fold_results.append(metrics)
                
                # Очистка места после обучения
                shutil.rmtree('./results/training_output', ignore_errors=True)
                shutil.rmtree('./logs', ignore_errors=True)


            if not fold_results:
                all_results.append({'ticker': ticker_name, 'status': 'failed_no_valid_folds'})
                continue

            avg_metrics = {k: np.mean([r[k] for r in fold_results]) for k in fold_results[0]}
            std_metrics = {f"{k}_std": np.std([r[k] for r in fold_results]) for k in ['accuracy', 'f1', 'auc']}
            
            result = {'ticker': ticker_name, 'status': 'success', 'n_folds': len(fold_results), **avg_metrics, **std_metrics}
            all_results.append(result)
            
            tqdm.write(f"✅ {ticker_name}: AUC = {result['auc']:.4f} ± {result['auc_std']:.4f}")

        # 4. Сохранение и анализ
        if all_results:
            self.save_results(all_results)
            self.analyze_results(all_results)
        
    def save_results(self, results: List[Dict]):
        """Сохраняет результаты в JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / f"experiment_results_{timestamp}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Результаты сохранены в: {filepath}")
    
    def analyze_results(self, results: List[Dict]):
        """Проводит анализ результатов и создает визуализации"""
        print("\n" + "="*80 + "\nАНАЛИЗ РЕЗУЛЬТАТОВ\n" + "="*80)
        
        successful = [r for r in results if r.get('status') == 'success']
        if not successful:
            print("❌ Нет успешных экспериментов для анализа.")
            return

        metrics_df = pd.DataFrame(successful).round(4)
        print(f"\n📈 Общая производительность (по {len(successful)} тикерам):")
        print(metrics_df[['ticker', 'auc', 'accuracy', 'f1']].sort_values('auc', ascending=False).to_string(index=False))
        
        print("\n" + "-"*60)
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            values = metrics_df[metric]
            print(f"{metric.upper():>12s}: Mean={values.mean():.4f}, Std={values.std():.4f}, Median={values.median():.4f}")
        
        self.create_visualizations(metrics_df)

    def create_visualizations(self, metrics_df: pd.DataFrame):
        """Создает и сохраняет графики"""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        sns.histplot(metrics_df['auc'], kde=True, ax=axes[0], bins=15, color='royalblue')
        axes[0].axvline(0.5, color='r', linestyle='--', label='Random Guess (0.5)')
        axes[0].set_title('Распределение AUC по всем тикерам')
        axes[0].set_xlabel('AUC Score')
        axes[0].legend()
        
        top_performers = metrics_df.nlargest(15, 'auc').sort_values('auc', ascending=True)
        axes[1].barh(top_performers['ticker'], top_performers['auc'], color='skyblue')
        axes[1].axvline(0.5, color='r', linestyle='--')
        axes[1].set_title('Топ-15 тикеров по средней AUC')
        axes[1].set_xlabel('AUC Score')
        
        plt.tight_layout()
        filepath = self.results_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filepath, dpi=300)
        print(f"\n📊 Визуализации сохранены в: {filepath}")
        plt.close()

def main():
    """Основная функция для запуска"""
    ## РЕКОМЕНДАЦИЯ: Вся конфигурация вынесена в один объект для удобства
    CONFIG = {
        'data_dir': 'Data/Tinkoff',
        'results_dir': 'results',
        'model_name': 'distilbert-base-uncased',
        'max_tickers': None, # None для всех тикеров, 20 для быстрого теста
        'max_folds': 3,
        'epochs': 2,
        'batch_size': 32,
        'train_size': 252, # 1 год
        'test_size': 21,   # 1 месяц
        'step_size': 21
    }
    
    experiment = MultiTickerExperiment(config=CONFIG)
    experiment.run()

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"✅ Найден GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ GPU не найден. Расчеты будут выполняться на CPU.")
    main()