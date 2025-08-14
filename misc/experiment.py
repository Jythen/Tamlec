import time
import shutil
import torch
from brutelogger import BruteLogger

from misc.utils import print_time, format_time_diff

if __name__ == '__main__':
    print("You should not call this directly, see inside the `configs` folder.")
    import sys; sys.exit()


# Set max number of threads used for one experiment
n_threads = 4
torch.set_num_threads(min(torch.get_num_threads(), n_threads))
torch.set_num_interop_threads(min(torch.get_num_interop_threads(), n_threads))

# Select the correct model/architecture according the method/algorithm picked
model_name = {
    
    'tamlec': 'tamlec',
   
}

# Set the task for the fewshot experiment
fewshot_task = {
    'eurlex': 120,
    'magcs': 6,
    'pubmed': 5,
}

# Batch size for hector and tamlec
batch_size = {
    
    'tamlec': {
        'eurlex': 16,
        'magcs': 32,
        'pubmed': 32,
        'oatopics': 64,
        'oaconcepts': 40,
    },
}

# Number of accumulations for hector and tamlec
accum_iter = {
  
    'tamlec': {
        'eurlex': 2,
        'magcs': 5,
        'pubmed': 2,
        'oatopics': 5,
        'oaconcepts': 2,
    },
}

# Epochs for the patience for each method
patience = {
   
    'tamlec': 3,
   
}



class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg['method'] = 'tamlec'

        # Create all paths
        # Output path set for the experiment
        output_path = self.cfg['output_path'] / self.cfg['exp_name'].stem
        predictions_path = output_path / 'predictions'
        preprocessed_data_path = self.cfg['dataset_path'] / 'preprocessed_data'
        self.cfg['paths'] = {
            'output': output_path,
            'dataset': self.cfg['dataset_path'],
            # Metrics
            'metrics': output_path / 'metrics.csv',
            'validation_metrics': output_path / f"{self.cfg['method']}_validation_metrics.csv",
            'test_metrics': output_path / f"{self.cfg['method']}_test_metrics.csv",
            'validation_level_metrics': output_path / f"{self.cfg['method']}_validation_level_metrics.csv",
            'test_level_metrics': output_path / f"{self.cfg['method']}_test_level_metrics.csv",
            # Saved models
            'model': output_path / 'model.pt',
            'state_dict': output_path / 'model_state_dict.pt',
            'model_finetuned': output_path / 'model_finetuned.pt',
            'state_dict_finetuned': output_path / 'model_state_dict_finetuned.pt',
            # Saved predictions
            'predictions_folder': predictions_path,
            'test_predictions': predictions_path / 'test_predictions.pt',
            'test_labels': predictions_path / 'test_labels.pt',
            'test_relevant_labels': predictions_path / 'relevant_labels.pt',
            'test_predictions_finetuned': predictions_path / 'test_predictions_finetuned.pt',
            'test_labels_finetuned': predictions_path / 'test_labels_finetuned.pt',
            # Saved pre-processed data
            'preprocessed_data': preprocessed_data_path,
            # Data for tamlec
            'taxos_tamlec': preprocessed_data_path / 'taxos_tamlec.pt',
            # Miscellaneous data
            'taxonomy': preprocessed_data_path / 'taxonomy.pt',
            'embeddings': preprocessed_data_path / 'embeddings.pt',
            'task_to_subroot': preprocessed_data_path / 'task_to_subroot.pt',
            'label_to_tasks': preprocessed_data_path / 'label_to_tasks.pt',
            'src_vocab': preprocessed_data_path / 'src_vocab.pt',
            'trg_vocab': preprocessed_data_path / 'trg_vocab.pt',
            'abstract_dict': preprocessed_data_path / 'abstract_dict.pt',
            'tasks_size': preprocessed_data_path / 'tasks_size.pt',
            'data': preprocessed_data_path / 'documents',
            'global_datasets': preprocessed_data_path / 'global_datasets.pt',
            'tasks_datasets': preprocessed_data_path / 'tasks_datasets.pt',
            'tokenizer': preprocessed_data_path / 'tokenizer.model',
            'vocabulary': preprocessed_data_path / 'tokenizer.vocab',
            'dataset_stats': preprocessed_data_path / 'dataset_stats',
            'drawn_tasks': preprocessed_data_path / 'dataset_stats' / 'drawn_tasks',
        }
        # Delete paths outside the newly-created dictionary
        del self.cfg['output_path']
        del self.cfg['dataset_path']

        # Create the folders
        self.cfg['paths']['output'].mkdir(exist_ok=True, parents=True)
        self.cfg['paths']['predictions_folder'].mkdir(exist_ok=True, parents=True)
        self.cfg['paths']['preprocessed_data'].mkdir(exist_ok=True, parents=True)
        self.cfg['paths']['data'].mkdir(exist_ok=True, parents=True)
        self.cfg['paths']['drawn_tasks'].mkdir(exist_ok=True, parents=True)
        # Save everything that is printed on the console to a log file
        BruteLogger.save_stdout_to_file(path=self.cfg['paths']['output'], fname="all_console.log")
        # Copy the config file in the output_directory
        shutil.copy2(self.cfg['exp_name'], self.cfg['paths']['output'])

        # Size of embedding space
        self.cfg['emb_dim'] = 300
        # Key used in metrics file to represent average of tasks
        self.cfg['all_tasks_key'] = 'global'
        self.cfg['model_name'] = model_name[self.cfg['method']]
        self.cfg['dataset'] = self.cfg['paths']['dataset'].name
       
        self.cfg['threshold'] = 0.5

        # Training loss function, optimizer and batch sizes
        self.cfg['loss_function'] = torch.nn.BCELoss()
        self.cfg['optimizer'] = torch.optim.AdamW
    
        try:
            self.cfg['batch_size_train'] = batch_size[self.cfg['method']][self.cfg['dataset']]
            self.cfg['batch_size_eval'] = batch_size[self.cfg['method']][self.cfg['dataset']]
            self.cfg['tamlec_params']['accum_iter'] = accum_iter[self.cfg['method']][self.cfg['dataset']]
        except KeyError:
            self.cfg['batch_size_train'] = 64
            self.cfg['batch_size_eval'] = 128
            self.cfg['tamlec_params']['accum_iter'] = 5
     
        self.cfg['patience'] = patience.get(self.cfg['method'], 5)

        # Pick a task for the fewshot experiment
        self.cfg['selected_task'] = fewshot_task.get(self.cfg['dataset'], 0)

       

        assert self.cfg['tokenization_mode'] in ['word', 'bpe', 'unigram'], f"{self.cfg['tokenization_mode']} should be ['word', 'bpe', 'unigram']"



    def main_run(self):
        start_time = time.perf_counter()

      
        
        from algorithms.tamlec import TamlecExp
        print_time(f"Starting TAMLEC experiment")
        experiment = TamlecExp(self.cfg)
        experiment.run()
       

        stop_time = time.perf_counter()
        print_time(f"Experiment ended in {format_time_diff(start_time, stop_time)}")
