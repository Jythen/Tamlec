from tqdm import tqdm
import torch
import numpy as np
import time

from datahandler.dataloading import load_data
from misc import metrics, utils


# Base experiment class that initializes the model, trains and evaluates it
# This base class needs to be used as parent class for the algorithms
class BaseExp:
    def __init__(self, cfg):
        self.cfg = cfg
        # Select correct torch device
        torch.cuda.set_device(self.cfg['device'])
        torch.cuda.device(self.cfg['device'])
        print(f"> Loading data...")
        self.dataloaders = load_data(self.cfg)
        self.taxonomy = self.cfg['taxonomy']
        # Monitor performance and stop after convergence
        self.early_stopping = utils.EarlyStopping(max_patience=self.cfg['patience'], cfg=self.cfg)
        # Save the metrics
        self.metrics_handler = {
            # Metrics during the training/fine-tuning part
            'training': utils.MetricsHandler(columns=['epoch', 'split', 'model', 'task', 'finetuning', 'metric', 'value'], output_path=self.cfg['paths']['metrics']),
            # Evaluation on validation and test sets
            'eval_validation': utils.MetricsHandler(columns=['model', 'task', 'finetuning', 'metric', 'value'], output_path=self.cfg['paths']['validation_metrics']),
            'eval_test': utils.MetricsHandler(columns=['model', 'task', 'finetuning', 'metric', 'value'], output_path=self.cfg['paths']['test_metrics']),
        }


    # These methods will be overridden in the children classes
    # Returns nothing
    def load_model(self):
        print(f">> load_model method should be overridden in the child class")
        import sys; sys.exit()

    # Returns the predictions
    def inference_eval(self, input_data, **kwargs):
        print(f">> inference_eval method should be overridden in the child class")
        import sys; sys.exit()
        # return predictions

    # Returns the train loss, and the gradients norms and their number
    def optimization_loop(self, input_data, labels, **kwargs):
        print(f">> optimization_loop method should be overridden in the child class")
        import sys; sys.exit()
        # return train_loss, (gradients_norms, gradient_lengths)

    # Returns nothing
    def run_init(self):
        print(f">> run_init method should be overridden in the child class")
        import sys; sys.exit()


    # To save memory, do no compute the gradients since we do not need them here
    @torch.no_grad()
    def eval_step(self, split, epoch, metrics_handler, verbose=False, save_pred=False, eval_finetuning=False, final_evaluation=False):
        # Return all metrics for all tasks and the global evaluation
        returned_dict = {}

        # 1. Global evaluation
        # Storages for global predictions
        all_predictions = []
        all_complete_labels = []
        all_relevant_labels = None

        for input_data, labels, column_indices in tqdm(self.dataloaders[f"global_{split}"], leave=False):
            # Send to device
            input_data = input_data.to(self.cfg['device'])
            predictions = self.inference_eval(input_data)
            # Add predictions and labels to the storage
            all_predictions.append(predictions.cpu())
            all_complete_labels.append(labels)
            # Same single tensor for all documents
            all_relevant_labels = column_indices[0]

        # Stack in one tensor
        all_predictions = torch.concat(all_predictions, dim=0)
        all_complete_labels = torch.concat(all_complete_labels, dim=0)

        # Save predictions, labels and relevant labels
        if save_pred and (not eval_finetuning):
            torch.save(all_predictions, self.cfg['paths'][f"{split}_predictions"])
            torch.save(all_complete_labels.bool(), self.cfg['paths'][f"{split}_labels"])
            # relevant labels are the same in the global evaluation
            torch.save(all_relevant_labels, self.cfg['paths'][f"{split}_relevant_labels"])
        if save_pred and eval_finetuning:
            torch.save(all_predictions, self.cfg['paths'][f"{split}_predictions_finetuned"])
            torch.save(all_complete_labels.bool(), self.cfg['paths'][f"{split}_labels_finetuned"])
            # relevant labels are the same in the global evaluation
            torch.save(all_relevant_labels, self.cfg['paths'][f"{split}_relevant_labels"])

        # Mask predictions and labels (i.e. remove the root label and roots of sub-taxonomies) and compute metrics
        filtered_predictions = all_predictions[:, all_relevant_labels]
        filtered_labels = all_complete_labels[:, all_relevant_labels]
        metrics_global, _n_docs = metrics.get_xml_metrics(
            filtered_predictions,
            filtered_labels,
            self.cfg['k_list'] if final_evaluation else self.cfg['k_list_eval_perf'],
            self.cfg['loss_function'],
            self.cfg['threshold'],
        )

        # Gather metrics for the global evaluation
        if verbose: print(f"> Results on the {split} set:")
        for metric_name, metric_value in metrics_global.items():
            if metrics_handler == 'training':
                new_row_dict = {
                    'epoch': epoch,
                    'split': split,
                    'model': self.cfg['method'],
                    'task': self.cfg['all_tasks_key'],
                    'finetuning': eval_finetuning,
                    'metric': metric_name,
                    'value': metric_value,
                }
            else:
                new_row_dict = {
                    'model': self.cfg['method'],
                    'task': self.cfg['all_tasks_key'],
                    'finetuning': eval_finetuning,
                    'metric': metric_name,
                    'value': metric_value,
                }
            self.metrics_handler[metrics_handler].add_row(new_row_dict)
            if verbose: print(f">> {metric_name} -> {metric_value}")
        # Add the computed metrics to the returned storage
        returned_dict[self.cfg['all_tasks_key']] = metrics_global

        # Compute the metrics per level at the final evaluation
        if final_evaluation:
            level_metrics_handler = utils.MetricsHandler(columns=['model', 'level', 'finetuning', 'metric', 'value'], output_path=self.cfg['paths'][f"{split}_level_metrics"])
            level_metrics = metrics.get_metrics_per_level(
                all_predictions,
                all_complete_labels,
                self.cfg['k_list'],
                self.taxonomy,
                self.cfg['loss_function'],
                self.cfg['threshold'],
            )
            # Aggregate level metrics
            for level, metrics_dict in level_metrics.items():
                print(f"\n>> Metrics at level {level}")
                for metric_name, metric_value in metrics_dict.items():
                    new_row_dict = {
                        'model': self.cfg['method'],
                        'level': level,
                        'finetuning': eval_finetuning,
                        'metric': metric_name,
                        'value': metric_value,
                    }
                    level_metrics_handler.add_row(new_row_dict)
                    print(f">>> {metric_name} -> {metric_value}")


        # 2. Tasks evaluation
        for task_id, (batched_input, batched_labels, column_indices) in enumerate(tqdm(self.dataloaders[f"tasks_{split}"], leave=False)):
            # Storages for this task
            task_predictions = []
            task_complete_labels = []

            for input_data, labels in zip(batched_input, batched_labels):
                # Predictions
                input_data = input_data.to(self.cfg['device'])
                predictions = self.inference_eval(input_data)
                # Add predictions and labels to the storage
                task_predictions.append(predictions.cpu())
                task_complete_labels.append(labels)

            # Stack all predictions and labels in one tensor
            task_predictions = torch.concat(task_predictions, dim=0)
            task_complete_labels = torch.concat(task_complete_labels, dim=0)

            # Mask predictions and label (i.e. retain labels that are in this task) and compute metrics
            task_predictions = task_predictions[:, column_indices]
            task_complete_labels = task_complete_labels[:, column_indices]
            metrics_task, n_docs = metrics.get_xml_metrics(
                task_predictions,
                task_complete_labels,
                self.cfg['k_list'] if final_evaluation else self.cfg['k_list_eval_perf'],
                self.cfg['loss_function'],
                self.cfg['threshold'],
            )

            # Gather metrics for the task
            for metric_name, metric_value in metrics_task.items():
                if metrics_handler == 'training':
                    new_row_dict = {
                        'epoch': epoch,
                        'split': split,
                        'model': self.cfg['method'],
                        'task': task_id,
                        'finetuning': eval_finetuning,
                        'metric': metric_name,
                        'value': metric_value,
                    }
                else:
                    new_row_dict = {
                        'model': self.cfg['method'],
                        'task': task_id,
                        'finetuning': eval_finetuning,
                        'metric': metric_name,
                        'value': metric_value,
                    }
                self.metrics_handler[metrics_handler].add_row(new_row_dict)
            # Save dict for this task
            returned_dict[task_id] = metrics_task

        return returned_dict


    def run(self):
        # 1. Train the model on the training set
        print(f"\n> Training the model")
        training = True
        epoch = 0
        split = 'train'
        finetuning = False
        start_train = time.perf_counter()
        # Miscellaneous initialization, if any, for the children classes
        self.run_init()

        # Evaluate the performance on the validation set before starting the training, i.e. at epoch 0
        before_training = self.eval_step(split='validation', epoch=epoch, metrics_handler='training', eval_finetuning=finetuning)
        print(f">> Epoch {epoch} | Validation loss -> {before_training[self.cfg['all_tasks_key']]['loss']} | prec@1 -> {before_training[self.cfg['all_tasks_key']]['precision@1']}")

        while training:
            epoch += 1
            # Storages for the training loss and gradients norms
            train_losses = []
            n_docs_per_batch = []
            grad_norms = []
            grad_lengths = []
            # Train on the batches
            for input_data, labels, _ in tqdm(self.dataloaders[f"global_{split}"], leave=False):
                # Send everything on device
                input_data = input_data.to(self.cfg['device'])
                labels = labels.to(self.cfg['device'])
                # Optimization loop
                train_loss, gradients_and_lengths = self.optimization_loop(input_data, labels)
                # Aggregate train loss and gradient norms
                train_losses.append(train_loss)
                n_docs_per_batch.append(len(input_data))
                grad_norms.append(gradients_and_lengths[0])
                grad_lengths.append(gradients_and_lengths[1])

            # Compute train loss for the whole epoch
            mean_train_loss = np.average(train_losses, weights=n_docs_per_batch)
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
            print(f">> Epoch {epoch} | Train loss -> {mean_train_loss:.3f} | Validation loss -> {val_metrics[self.cfg['all_tasks_key']]['loss']:.3f} | Gradients norm -> {np.average(grad_norms, weights=grad_lengths):.6f}")

            # Check if we stop training or not
            training = self.early_stopping.checkpoint(self.model, val_metrics[self.cfg['all_tasks_key']], epoch)

        stop_train = time.perf_counter()
        print(f"> End training at epoch {epoch} after {utils.format_time_diff(start_train, stop_train)}")

        # Evaluate the best model
        splits_to_eval = ['validation', 'test']
        # Reload best model
        self.load_model()
        for split in splits_to_eval:
            print(f"\n> Evaluating the best model on the {split} set...")
            self.eval_step(split=split, epoch='best', verbose=True, metrics_handler=f"eval_{split}", save_pred=split=='test', eval_finetuning=finetuning, final_evaluation=True)


        # 2. In few-shot experiment, fine-tune the task that has not been trained
        if self.cfg['fewshot_exp']:
            print(f"\n> Fine-tuning on task {self.cfg['selected_task']}")
            # Reload best model
            self.load_model()
            training = True
            epoch_finetuning = epoch
            split = 'train'
            finetuning = True
            # Monitor the progression during the fine-tuning
            finetune_patience = utils.EarlyStopping(max_patience=self.cfg['patience'], cfg=self.cfg, finetuning=True)
            start_finetune = time.perf_counter()

            while training:
                epoch_finetuning += 1
                # Storages for the training loss and gradients norms
                train_losses = []
                n_docs_per_task = []
                grad_norms = []
                grad_lengths = []
                for task_id, (batched_input, batched_labels, _) in enumerate(tqdm(self.dataloaders[f"tasks_{split}"], leave=False)):
                    # Train only on the selected task
                    if task_id != self.cfg['selected_task']: continue
                    batch_loss = []
                    n_docs_per_batch = []

                    # Train on batches
                    for input_data, labels in zip(batched_input, batched_labels):
                        # Send everything on device
                        input_data = input_data.to(self.cfg['device'])
                        labels = labels.to(self.cfg['device'])
                        # Optimization loop
                        train_loss, gradients_and_lengths = self.optimization_loop(input_data, labels)
                        # Aggregate train loss and gradient norms
                        batch_loss.append(train_loss)
                        n_docs_per_batch.append(len(input_data))
                        grad_norms.append(gradients_and_lengths[0])
                        grad_lengths.append(gradients_and_lengths[1])

                    # Save the loss of the task
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
                print(f">> Epoch {epoch_finetuning} | Train loss -> {mean_train_loss:.3f} | Validation loss -> {val_metrics[self.cfg['all_tasks_key']]['loss']:.3f} | Gradients norm -> {np.average(grad_norms, weights=grad_lengths):.6f}")

                # Check if we stop fine_tuning or not
                training = finetune_patience.checkpoint(self.model, val_metrics[self.cfg['selected_task']], epoch_finetuning)

            stop_finetune = time.perf_counter()
            print(f"> End fine-tuning at epoch {epoch_finetuning} after {utils.format_time_diff(start_finetune, stop_finetune)}")

            # Evaluate the best model
            splits_to_eval = ['validation', 'test']
            # Reload best model
            self.load_model()
            for split in splits_to_eval:
                print(f"\n> Evaluating the best fine-tuned model on the {split} set...")
                self.eval_step(split=split, epoch='best', verbose=True, metrics_handler=f"eval_{split}", save_pred=split=='test', eval_finetuning=finetuning, final_evaluation=True)

        # Print the config so we have it in case of logging or in the terminal
        utils.print_config(self.cfg)
