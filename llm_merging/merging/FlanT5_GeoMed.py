import torch 
from llm_merging.merging.Merges import Merges
from peft import get_peft_model, set_peft_model_state_dict
from typing import List, Tuple, Dict

class FlanT5GeoMed(Merges):
    def __init__(self, name: str):
        super().__init__(name)

        # List of models to load for the merge 
        self.list_models: List[Tuple[str, str]] = [
            ("lorahub/flan_t5_xl-wiki_qa_Is_This_True_", "30a1ee2f857196c1eb996d854548cc19f45ac642"), 
            ("lorahub/flan_t5_xl-anli_r2","ea7872e79fddc6e9df57b88c429bdb283b414bea"),
            ("lorahub/flan_t5_xl-kilt_tasks_hotpotqa_complex_question", "27d014366bec1c5333ba2e2fae966b7de3c02df1"),
            ("lorahub/flan_t5_xl-web_questions_question_answer", "37701f6f673974308517151387182f42271a2eab"),
            ("lorahub/flan_t5_xl-gem_e2e_nlg", "04e25c5739d151e42916b262cb0ee900aa854816"),
            ("lorahub/flan_t5_xl-wiki_hop_original_explain_relation", "d6bdec80c60d55db0b7125f8ca0d02871ab3ab34"),
            ("lorahub/flan_t5_xl-dbpedia_14_given_list_what_category_does_the_paragraph_belong_to","883db61b41a3a9e8716f5391d782f653fd9d693b"),
            ("lorahub/flan_t5_xl-wiki_bio_comprehension","9d06f885dbbbe69327203b299193873ea281522c"),
            ("lorahub/flan_t5_xl-wiki_bio_guess_person","e8998f9f0fad7aef94408c4741e7fbe2ff11f79d"),
            ("lorahub/flan_t5_xl-wiki_bio_who","c081565f0d3e3aa251fa9d44fc6678d70cc9e20f"),
            ("lorahub/flan_t5_xl-wiki_qa_Topic_Prediction_Question_Only","cc024699f37aee24e72cd28a596dbf3451a93484"),
            ("lorahub/flan_t5_xl-gem_web_nlg_en","8043f44956456dffb6cc5e07bc59bffdf618ac97"),            
        ]

        # Hyperparameters 
        self.base_model_name: str = "google/flan-t5-xl"
        self.base_model_revision_id: str = "7d6315df2c2fb742f0f5b556879d730926ca9001"
        self.is_peft: bool = True
        self.max_seq_len: int = 512
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Architecture must match base model. 
        self.architecture: str = "encoder_decoder"

        # Loaded models and configs 
        self.loaded_models: Dict[str, Dict[str, torch.Tensor]] = {}
        self.loaded_configs: Dict[str, any] = {}

        # Merged model parameters
        self.merged_model: Dict[str, torch.Tensor] = {}
    

    def merge(self):
        ''' Load HuggingFace checkpoints and configs '''
        super()._load_huggingface_models_and_configs()

        ''' Load base model and tokenizer '''
        self._load_base_model()  # Load the pre-trained base model
        self._load_tokenizer()
        # print("Base model and tokenizer gets loaded properly")

        # Get base model's parameters
        base_model_parameters = self.base_model.state_dict()
        # print("Base Model Parameters (before merging):")
        base_param = {}
        for param_name, param in base_model_parameters.items():
            base_param[param_name] = param.to(self.device)

        # Merge using the geometric median
        all_models = list(self.loaded_models.values())
        all_parameter_names = all_models[0].keys()

        # Apply geometric median for each parameter
        for parameter_name in all_parameter_names:
            task_vectors = [model_params[parameter_name].to(self.device) for model_params in all_models]
            merged_parameter = self.compute_geometric_median(task_vectors)

            # Store merged parameters
            self.merged_model[parameter_name] = merged_parameter

        # Apply the geometric median to the base model
        # self.apply_geometric_median(torch.cat([param.view(-1) for param in self.merged_model.values()]))

        # Set the model to evaluation mode
        self.base_model.eval()

        ''' Load merged model into base model '''
        huggingface_config = list(self.loaded_configs.values())[0]
        if huggingface_config is not None:
            self.base_model = get_peft_model(self.base_model, huggingface_config)
            set_peft_model_state_dict(self.base_model, self.merged_model)
        else:
            self.base_model.load_state_dict(self.merged_model)

        # Set the model to evaluation mode
        self.base_model.eval()

        # print("Geometric median applied to the base model successfully.")

        return self.base_model
    
    def compute_geometric_median(self, task_vectors: List[torch.Tensor], eps: float = 1e-8, max_iter: int = 300) -> torch.Tensor:
        """
        Compute the geometric median of the given task vectors using Weiszfeld's algorithm.
        Args:
            task_vectors (List[torch.Tensor]): List of 1D tensors representing task vectors.
            eps (float): Convergence threshold.
            max_iter (int): Maximum number of iterations.
        Returns:
            torch.Tensor: Geometric median as a 1D tensor.
        """
        if not task_vectors:
            raise ValueError("No task vectors provided for geometric median computation.")

        # Initialize the median as the mean of the task vectors
        median = torch.mean(torch.stack(task_vectors), dim=0)
        median = median.to(self.device)

        for iteration in range(max_iter):
            distances = torch.stack([torch.norm(tv - median) for tv in task_vectors])
            # Avoid division by zero
            distances[distances < 1e-10] = 1e-10
            weights = 1.0 / distances
            weights_sum = torch.sum(weights)
            if weights_sum == 0:
                # print("Weights sum to zero during geometric median computation.")
                break
            weighted_sum = torch.stack([w * tv for w, tv in zip(weights, task_vectors)]).sum(dim=0)
            new_median = weighted_sum / weights_sum

            shift = torch.norm(new_median - median)
            # print(f"Iteration {iteration}: shift={shift.item()}")

            if shift < eps:
                # print(f"Converged after {iteration} iterations.")
                return new_median

            median = new_median

        # print(f"Geometric median did not converge after {max_iter} iterations. Returning the last estimate.")
        return median
