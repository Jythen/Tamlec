import torch
import numpy as np
import nltk
from tqdm import tqdm
import time

from Tamlec.tamlec import Tamlec
from Tamlec.prediction_handler import CustomXMLHolder
from datahandler.dataloading import load_data
from misc import metrics, utils


class TamlecExp:
    def __init__(self, cfg):
        # Download from nltk
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')

        self.cfg = cfg
        torch.cuda.set_device(self.cfg['device'])
        torch.cuda.device(self.cfg['device'])
        print(f"> Loading data...")
        self.dataloaders = load_data(self.cfg)
        print(f"> Loading model...")
        self.model = Tamlec(
            src_vocab=self.cfg['tamlec_params']['src_vocab'],
            tgt_vocab=self.cfg['tamlec_params']['trg_vocab'],
            path_to_glove=".vector_cache/glove.840B.300d.gensim",
            abstract_dict=self.cfg['tamlec_params']['abstract_dict'],
            taxonomies=self.cfg['tamlec_params']['taxos_tamlec'],
            width_adaptive=self.cfg['tamlec_params']['width_adaptive'],
            decoder_adaptative=self.cfg['tamlec_params']['decoder_adaptative'],
            tasks_size=self.cfg['tamlec_params']['tasks_size'],
            gpu_target=self.cfg['device'],
            with_bias=self.cfg['tamlec_params']['with_bias'],
            Number_src_blocs=6,
            Number_tgt_blocs=6,
            dim_src_embedding=300,
            dim_tgt_embedding=600,
            dim_feed_forward=2048,
            number_of_heads=12,
            dropout=0.1,
            learning_rate=self.cfg['learning_rate'],
            beta1=0.9,
            beta2=0.99,
            epsilon=1e-8,
            weight_decay=0.01,
            gamma=.99998,
            accum_iter=self.cfg['tamlec_params']['accum_iter'],
            # 0.0 < x <= 0.1
            loss_smoothing=self.cfg['tamlec_params']['loss_smoothing'],
            max_padding_document=self.cfg['seq_length'],
            max_number_of_labels=20,
        )
        if self.cfg['tamlec_params']['freeze']:
            self.early_stopping = utils.AdaptivePatience(max_patience=self.cfg['patience'], cfg=self.cfg, n_tasks=len(self.dataloaders['tasks_validation']))
        else:
            self.early_stopping = utils.EarlyStopping(max_patience=self.cfg['patience'], cfg=self.cfg)
        self.metrics_handler = {
            # Metrics during the training/fine-tuning part
            'training': utils.MetricsHandler(columns=['epoch', 'split', 'model', 'task', 'finetuning', 'metric', 'value'], output_path=self.cfg['paths']['metrics']),
            # Evaluation on validation and test sets
            'eval_validation': utils.MetricsHandler(columns=['model', 'task', 'finetuning', 'metric', 'value'], output_path=self.cfg['paths']['validation_metrics']),
            'eval_test': utils.MetricsHandler(columns=['model', 'task', 'finetuning', 'metric', 'value'], output_path=self.cfg['paths']['test_metrics']),
        }


    # To save memory, do no compute the gradients since we do not need them here
    @torch.no_grad()
    def eval_step(self, split, epoch, metrics_handler, eval_finetuning):
        # Set model to eval() modifies the behavior of certain layers e.g. dropout, normalization layers
        self.model.eval()
        # Store the mean metrics for this evaluation step
        mean_metrics = {}
        # Return all metrics for all tasks, and the average
        returned_dict = {self.cfg['all_tasks_key']: {}}
        n_docs_per_task = []

        for task_id, (batched_input, batched_labels, _) in enumerate(tqdm(self.dataloaders[f"tasks_{split}"], leave=False)):
            losses = []
            all_precisions_at = {k: [] for k in self.cfg['k_list_eval_perf']}
            total_masses = []
            n_docs_per_batch = []
            for input_data, labels in zip(batched_input, batched_labels):
                # Test on batch, aggregate metrics for this batch
                loss, precisions, total_mass = self.model.eval_batch(documents_tokens=input_data, labels_tokens=labels, task_id=task_id)
                losses.append(loss.cpu().item())
                for k in self.cfg['k_list_eval_perf']:
                    all_precisions_at[k].append(precisions[k])
                total_masses.append(total_mass)
                n_docs_per_batch.append(len(input_data))

            # Gather metrics for the task
            metrics_dict = {
                'loss': np.average(losses, weights=n_docs_per_batch),
                'mass': np.average(total_masses, weights=n_docs_per_batch),
            }
            for k in self.cfg['k_list_eval_perf']: metrics_dict[f"precision@{k}"] = np.average(all_precisions_at[k], weights=n_docs_per_batch)

            # Save metrics
            returned_dict[task_id] = metrics_dict
            n_docs_per_task.append(np.sum(n_docs_per_batch))
            for metric_name, metric_value in metrics_dict.items():
                new_row_dict = {
                    'epoch': epoch,
                    'split': split,
                    'model': self.cfg['method'],
                    'task': task_id,
                    'finetuning': eval_finetuning,
                    'metric': metric_name,
                    'value': metric_value,
                }
                self.metrics_handler[metrics_handler].add_row(new_row_dict)
                # Add the computed value to the mean storage
                try:
                    mean_metrics[metric_name].append(metric_value)
                except KeyError:
                    mean_metrics[metric_name] = [metric_value]

        # Average over the tasks and save metrics
        for metric_name, metric_values in mean_metrics.items():
            mean_metric = np.average(metric_values, weights=n_docs_per_task)
            new_row_dict = {
                'epoch': epoch,
                'split': split,
                'model': self.cfg['method'],
                'task': self.cfg['all_tasks_key'],
                'finetuning': eval_finetuning,
                'metric': metric_name,
                'value': mean_metric,
            }
            self.metrics_handler[metrics_handler].add_row(new_row_dict)
            returned_dict[self.cfg['all_tasks_key']][metric_name] = mean_metric

        return returned_dict


    def run(self):
        # 1. Train the model on the training set
        print(f"\n> Training the model")
        training = True
        epoch = 0
        split = 'train'
        finetuning = False
        start_train = time.perf_counter()

        # Evaluate the performance on the validation set before training
        before_training = self.eval_step(split='validation', epoch=epoch, metrics_handler='training', eval_finetuning=finetuning)
        print(f">> Epoch {epoch} | Validation loss -> {before_training[self.cfg['all_tasks_key']]['loss']} | prec@1 -> {before_training[self.cfg['all_tasks_key']]['precision@1']}")

        while training:
            epoch += 1
            train_losses = []
            n_docs_per_task = []
            # Model in training mode, e.g. activate dropout if specified
            self.model.train()
            for task_id, (batched_input, batched_labels, _) in enumerate(tqdm(self.dataloaders[f"tasks_{split}"], leave=False)):
                batch_loss = []
                n_docs_per_batch = []
                # In fewshot, skip the selected task
                if self.cfg['fewshot_exp'] and task_id == self.cfg['selected_task']: continue

                # Train on batches
                for input_data, labels in zip(batched_input, batched_labels):
                    loss = self.model.train_on_batch(documents_tokens=input_data, labels_tokens=labels, task_id=task_id)
                    batch_loss.append(loss.cpu().item())
                    n_docs_per_batch.append(len(input_data))

                # Save the loss
                train_loss = np.average(batch_loss, weights=n_docs_per_batch)
                train_losses.append(train_loss)
                n_docs_per_task.append(np.sum(n_docs_per_batch))
                new_row_dict = {
                    'epoch': epoch,
                    'split': split,
                    'model': self.cfg['method'],
                    'task': task_id,
                    'finetuning': finetuning,
                    'metric': 'loss',
                    'value': train_loss,
                }
                self.metrics_handler['training'].add_row(new_row_dict)

            # Compute train loss for the whole epoch
            mean_train_loss = np.average(train_losses, weights=n_docs_per_task)
            new_row_dict = {
                'epoch': epoch,
                'split': split,
                'model': self.cfg['method'],
                'task': self.cfg['all_tasks_key'],
                'finetuning': finetuning,
                'metric': 'loss',
                'value': mean_train_loss,
            }
            self.metrics_handler['training'].add_row(new_row_dict)

            # Evaluate the performance on the validation set
            val_metrics = self.eval_step(split='validation', epoch=epoch, metrics_handler='training', eval_finetuning=finetuning)
            print(f">> Epoch {epoch} | Train loss -> {mean_train_loss:.3f} | Validation loss -> {val_metrics[self.cfg['all_tasks_key']]['loss']:.3f}")

            # Check if we stop training or not
            if self.cfg['tamlec_params']['freeze']:
                training = self.early_stopping.global_checkpoint(self.model, val_metrics, epoch)
            else:
                training = self.early_stopping.checkpoint(self.model, val_metrics[self.cfg['all_tasks_key']], epoch)

        # If in freeze mode, second part is to train all tasks that have not converged yet
        if self.cfg['tamlec_params']['freeze']:
            # In fewshot experiment anyway mark that the selected task has converged
            if self.cfg['fewshot_exp']: self.early_stopping.mark_task_done(self.cfg['selected_task'])
            if len(self.early_stopping.tasks_to_complete) == 0:
                print(f"\n All tasks already converged")
                training = False
            else:
                print(f"\n> Freeze shared parameters and continue training on {len(self.early_stopping.tasks_to_complete)} tasks: {self.early_stopping.tasks_to_complete}")
                training = True
            # Freeze parameters shared in all tasks
            self.model.freeze()
            while training:
                epoch += 1
                train_losses = []
                n_docs_per_task = []
                # Model in training mode, e.g. activate dropout if specified
                self.model.train()
                for task_id, (batched_input, batched_labels, _) in enumerate(tqdm(self.dataloaders[f"tasks_{split}"], leave=False)):
                    # Train only tasks that have not converged yet
                    if task_id in self.early_stopping.tasks_completed: continue
                    batch_loss = []
                    n_docs_per_batch = []

                    # Train on batches
                    for input_data, labels in zip(batched_input, batched_labels):
                        loss = self.model.train_on_batch(documents_tokens=input_data, labels_tokens=labels, task_id=task_id)
                        batch_loss.append(loss.cpu().item())
                        n_docs_per_batch.append(len(input_data))

                    # Save the loss
                    train_loss = np.average(batch_loss, weights=n_docs_per_batch)
                    train_losses.append(train_loss)
                    n_docs_per_task.append(np.sum(n_docs_per_batch))
                    new_row_dict = {
                        'epoch': epoch,
                        'split': split,
                        'model': self.cfg['method'],
                        'task': task_id,
                        'finetuning': finetuning,
                        'metric': 'loss',
                        'value': train_loss,
                    }
                    self.metrics_handler['training'].add_row(new_row_dict)

                # Compute train loss for the whole epoch
                mean_train_loss = np.average(train_losses, weights=n_docs_per_task)
                new_row_dict = {
                    'epoch': epoch,
                    'split': split,
                    'model': self.cfg['method'],
                    'task': self.cfg['all_tasks_key'],
                    'finetuning': finetuning,
                    'metric': 'loss',
                    'value': mean_train_loss,
                }
                self.metrics_handler['training'].add_row(new_row_dict)

                # Evaluate the performance on the validation set
                val_metrics = self.eval_step(split='validation', epoch=epoch, metrics_handler='training', eval_finetuning=finetuning)
                print(f">> Epoch {epoch} | Train loss -> {mean_train_loss:.3f} | Validation loss -> {val_metrics[self.cfg['all_tasks_key']]['loss']:.3f}")

                # Check if we stop training or not
                training = self.early_stopping.tasks_checkpoint(self.model, val_metrics, epoch)
                # Anyway save the model since only specific weights are updated
                if finetuning:
                    torch.save(self.model, self.cfg['paths']['model_finetuned'])
                else:
                    torch.save(self.model, self.cfg['paths']['model'])

        stop_train = time.perf_counter()
        print(f"> End training at epoch {epoch} after {utils.format_time_diff(start_train, stop_train)}")

        # Evaluate the best model
        splits_to_eval = ['validation', 'test']
        # Reload best model
        with open(self.cfg['paths']['model'], "rb") as f:
            self.model = torch.load(f, map_location=self.cfg['device'])
        for split in splits_to_eval:
            print(f"\n> Evaluating the best model on the {split} set...")
            self.final_evaluation(split=split, metrics_handler=f"eval_{split}", save_pred=split=='test', eval_finetuning=finetuning)


        # 2. In few-shot experiment, fine-tune the task that has not been trained
        if self.cfg['fewshot_exp']:
            print(f"\n> Fine-tuning on task {self.cfg['selected_task']}")
            # Reload best model and setup optimizer
            with open(self.cfg['paths']['model'], "rb") as f:
                self.model = torch.load(f, map_location=self.cfg['device'])
            self.model.get_optimizer()

            training = True
            epoch_finetuning = epoch
            split = 'train'
            finetuning = True
            finetune_patience = utils.EarlyStopping(max_patience=self.cfg['patience'], cfg=self.cfg, finetuning=True)
            start_finetune = time.perf_counter()

            while training:
                epoch_finetuning += 1
                # Loss for all tasks, and number of documents per task for weighted average
                train_losses = []
                n_docs_per_task = []
                # Model in training mode, e.g. activate dropout if specified
                self.model.train()
                for task_id, (batched_input, batched_labels, _) in enumerate(tqdm(self.dataloaders[f"tasks_{split}"], leave=False)):
                    # Train only on the selected task
                    if task_id != self.cfg['selected_task']: continue
                    batch_loss = []
                    n_docs_per_batch = []

                    # Train on batches
                    for input_data, labels in zip(batched_input, batched_labels):
                        loss = self.model.train_on_batch(documents_tokens=input_data, labels_tokens=labels, task_id=task_id)
                        batch_loss.append(loss.cpu().item())
                        n_docs_per_batch.append(len(input_data))

                    # Save the loss
                    train_loss = np.average(batch_loss, weights=n_docs_per_batch)
                    train_losses.append(train_loss)
                    n_docs_per_task.append(np.sum(n_docs_per_batch))
                    new_row_dict = {
                        'epoch': epoch_finetuning,
                        'split': split,
                        'model': self.cfg['method'],
                        'task': task_id,
                        'finetuning': finetuning,
                        'metric': 'loss',
                        'value': train_loss,
                    }
                    self.metrics_handler['training'].add_row(new_row_dict)

                # Compute train loss for the whole epoch
                mean_train_loss = np.average(train_losses, weights=n_docs_per_task)
                new_row_dict = {
                    'epoch': epoch_finetuning,
                    'split': split,
                    'model': self.cfg['method'],
                    'task': self.cfg['all_tasks_key'],
                    'finetuning': finetuning,
                    'metric': 'loss',
                    'value': mean_train_loss,
                }
                self.metrics_handler['training'].add_row(new_row_dict)

                # Evaluate the performance on the validation set
                val_metrics = self.eval_step(split='validation', epoch=epoch_finetuning, metrics_handler='training', eval_finetuning=finetuning)
                print(f">> Epoch {epoch_finetuning} | Train loss -> {mean_train_loss:.3f} | Validation loss -> {val_metrics[self.cfg['all_tasks_key']]['loss']:.3f}")

                # Check if we stop fine_tuning or not
                training = finetune_patience.checkpoint(self.model, val_metrics[self.cfg['selected_task']], epoch_finetuning)

            stop_finetune = time.perf_counter()
            print(f"> End fine-tuning at epoch {epoch_finetuning} after {utils.format_time_diff(start_finetune, stop_finetune)}")

            # Evaluate the best model
            splits_to_eval = ['validation', 'test']
            # Reload best model
            with open(self.cfg['paths']['model_finetuned'], "rb") as f:
                self.model = torch.load(f, map_location=self.cfg['device'])
            for split in splits_to_eval:
                print(f"\n> Evaluating the best fine-tuned model on the {split} set...")
                self.final_evaluation(split=split, metrics_handler=f"eval_{split}", save_pred=split=='test', eval_finetuning=finetuning)

        utils.print_config(self.cfg)


    @torch.no_grad()
    def final_evaluation(self, split, metrics_handler, save_pred, eval_finetuning):
        # Initialize all storages
        all_predictions = []
        all_complete_labels = []
        all_relevant_classes = []
        all_metrics = {}
        n_docs_per_task = {}

        # Set model to eval() modifies the behavior of certain layers e.g. dropout, normalization layers
        self.model.eval()

        for task_id, (batched_input, batched_labels, single_relevant_classes) in enumerate(tqdm(self.dataloaders[f"tasks_{split}"], leave=False)):
            # Aggregate batched data (3D) to un-batched data (2D)
            input_data = torch.vstack(batched_input)
            labels = []
            relevant_classes = []
            for labels_in_batch in batched_labels:
                for label_sample in labels_in_batch:
                    labels.append(label_sample)
                    relevant_classes.append(single_relevant_classes)

            # Compute predictions
            xml = CustomXMLHolder(text_batch=input_data, task_id=task_id, beam_parameter=10, tamlec=self.model)
            predictions, _scores = xml.run_predictions(batch_size=128)

            # To save stuff later
            all_predictions.append(predictions)
            all_complete_labels.append(labels)
            all_relevant_classes.append(relevant_classes)

            # Compute the metrics
            all_results = []
            for star_input in zip(predictions, labels, relevant_classes):
                all_results.append(self.doc_metrics(*star_input))

            # Aggregate predictions
            n_docs_per_metric = {}
            task_metrics = {}
            for metric_dict in all_results:
                for metric_name, metric_value in metric_dict.items():
                    try:
                        task_metrics[metric_name].append(metric_value)
                        n_docs_per_metric[metric_name] += 1 
                    except KeyError:
                        task_metrics[metric_name] = [metric_value]
                        n_docs_per_metric[metric_name] = 1

            # Average over documents and save for global aggregation
            for metric_name, metric_values in task_metrics.items():
                mean_val = np.mean(metric_values)
                new_row_dict = {
                    'model': self.cfg['method'],
                    'task': task_id,
                    'finetuning': eval_finetuning,
                    'metric': metric_name,
                    'value': mean_val,
                }
                self.metrics_handler[f"eval_{split}"].add_row(new_row_dict)
                try:
                    all_metrics[metric_name].append(mean_val)
                    n_docs_per_task[metric_name].append(n_docs_per_metric[metric_name])
                except KeyError:
                    all_metrics[metric_name] = [mean_val]
                    n_docs_per_task[metric_name] = [n_docs_per_metric[metric_name]]

        # Compute the global metrics (weighted average over tasks)
        print(f"> Results on the {split} set")
        for metric_name in all_metrics.keys():
            mean_val = np.average(all_metrics[metric_name], weights=n_docs_per_task[metric_name])
            new_row_dict = {
                'model': self.cfg['method'],
                'task': self.cfg['all_tasks_key'],
                'finetuning': eval_finetuning,
                'metric': metric_name,
                'value': mean_val,
            }
            self.metrics_handler[metrics_handler].add_row(new_row_dict)
            print(f">> {metric_name} -> {mean_val}")

        if save_pred:
            torch.save(all_relevant_classes, self.cfg['paths'][f"{split}_relevant_labels"])
            if eval_finetuning:
                torch.save(all_predictions, self.cfg['paths'][f"{split}_predictions_finetuned"])
                torch.save(all_complete_labels, self.cfg['paths'][f"{split}_labels_finetuned"])
            else:
                torch.save(all_predictions, self.cfg['paths'][f"{split}_predictions"])
                torch.save(all_complete_labels, self.cfg['paths'][f"{split}_labels"])


    def doc_metrics(self, prediction, label_list, relevant_classes):
        metrics_document = {}
        # Take only predictions from labels that are appearing in this task
        doc_pred = []
        for class_idx in prediction:
            if class_idx in relevant_classes:
                doc_pred.append(class_idx)
        # Also filter only relevant labels
        filtered_labels = [label for label in label_list if label in relevant_classes]

        # Compute the metrics
        for k in self.cfg['k_list']:
            if len(filtered_labels) < k: continue
            # Compute precision@k
            precision = 0
            for pred in doc_pred[:k]:
                if pred in filtered_labels:
                    precision += 1
            metrics_document[f"precision@{k}"] = precision / k

            # Compute rankedsum@k
            ranked = []
            for rank, pred in enumerate(doc_pred):
                if pred in filtered_labels:
                    ranked.append(rank+1)
            metrics_document[f"rankedsum@{k}"] = ranked[k-1]

            # Compute ndcg@k
            metrics_document[f"ndcg@{k}"] = metrics.ndcg_k_hector(doc_pred, filtered_labels, k=k)

        return metrics_document
