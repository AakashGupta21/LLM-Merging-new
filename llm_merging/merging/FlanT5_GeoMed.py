import torch
from llm_merging.merging.Merges import Merges
from typing import List, Tuple, Dict, Any
from peft import get_peft_model, set_peft_model_state_dict

class FlanT5GeoMed(Merges):
    def _init_(self, name: str):
        super().__init__(name)

        # List of models to load for the merge
        self.list_models: List[Tuple[str, str]] = [
            # Text Classification
            # Models for classifying text into categories
            ("lorahub/flan_t5_xl-dbpedia_14_given_list_what_category_does_the_paragraph_belong_to", "883db61b41a3a9e8716f5391d782f653fd9d693b"),
            ("lorahub/flan_t5_xl-wiki_qa_Topic_Prediction_Question_Only", "cc024699f37aee24e72cd28a596dbf3451a93484"),

            # Question Answering
            # Models for answering questions based on provided context
            ("lorahub/flan_t5_xl-anli_r2", "ea7872e79fddc6e9df57b88c429bdb283b414bea"),
            ("lorahub/flan_t5_xl-web_questions_question_answer", "37701f6f673974308517151387182f42271a2eab"),
            ("lorahub/flan_t5_xl-duorc_SelfRC_question_answering", "b56b5b0b72a0a4b90b120833ff466aa7ef85dd84"),
            ("lorahub/flan_t5_xl-adversarial_qa_dbert_question_context_answer", "a935c63c0c7deaca77f437efd3425192a88dd90e"),


            # Text Generation
            # Model for end-to-end text generation tasks
            ("lorahub/flan_t5_xl-gem_e2e_nlg", "04e25c5739d151e42916b262cb0ee900aa854816"),

            # Text2Text Generation
            # Models for generating text based on input text
            ("lorahub/flan_t5_xl-wiki_hop_original_explain_relation", "d6bdec80c60d55db0b7125f8ca0d02871ab3ab34"),
            ("lorahub/flan_t5_xl-duorc_SelfRC_title_generation", "17653e0c744bb1453f93b816d1eb140d991be6a4"),

            # Sentence Similarity
            # Models for assessing the similarity between sentences
            ("lorahub/flan_t5_xl-glue_mrpc", "292a6f0c2dec34a9faa143b37dc734eee14c860a"),
            ("lorahub/flan_t5_xl-glue_cola", "7fef5d273d145e26b07762b43abcbaa83874dc23"),

            # Additional Tasks
            # Models for various comprehension and understanding tasks
            # Comprehension Tasks
            ("lorahub/flan_t5_xl-wiki_bio_comprehension", "9d06f885dbbbe69327203b299193873ea281522c"),
            ("lorahub/flan_t5_xl-wiki_bio_key_content", "f98ee1718a9ce23446671023a60fb05a57f5e9d3"),
            ("lorahub/flan_t5_xl-wiki_bio_guess_person", "e8998f9f0fad7aef94408c4741e7fbe2ff11f79d"),
            ("lorahub/flan_t5_xl-wiki_bio_who", "c081565f0d3e3aa251fa9d44fc6678d70cc9e20f"),

            # Search and Retrieval Tasks
            ("lorahub/flan_t5_xl-wiki_qa_found_on_google", "cb5c59ee688f22e0314968e2a0c1bee692e66c27"),

            # Natural Language Generation Tasks
            ("lorahub/flan_t5_xl-gem_web_nlg_en", "8043f44956456dffb6cc5e07bc59bffdf618ac97"),

            # Paraphrasing and Extraction Tasks
            ("lorahub/flan_t5_xl-duorc_ParaphraseRC_extract_answer", "c008dacf47c7836a0bcd2d4c47cd27923d2cda1e"),
            ("lorahub/flan_t5_xl-duorc_SelfRC_extract_answer", "377a71b7c71099688c836d7417eb9cfc0c33f6b5"),

            # Process Understanding Tasks
            ("lorahub/flan_t5_xl-wiqa_what_might_be_the_last_step_of_the_process", "fea37d25cf4eb8d81a85fc3296e7781fc8ea10db"),
        ]
        #hyperparameters
        self.base_model_name: str = "google/flan-t5-xl"
        self.base_model_revision_id: str = "7d6315df2c2fb742f0f5b556879d730926ca9001"
        self.is_peft: bool = True
        self.max_seq_len: int = 512
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.architecture: str = "encoder_decoder"

        self.loaded_models: Dict[str, Dict[str, torch.Tensor]] = {}
        self.loaded_configs: Dict[str, Any] = {}
        self.merged_model: Dict[str, torch.Tensor] = {}

    def merge(self):
        ''' Load HuggingFace checkpoints and configs '''
        super()._load_huggingface_models_and_configs()

        ''' Load base model and tokenizer '''
        self._load_base_model()
        self._load_tokenizer()

        #print("Base model and tokenizer loaded properly.")

        if not self.loaded_models:
            raise ValueError("No models loaded for merging.")

        base_model_params = self.base_model.state_dict()
        #print(type(base_model_params))

        # Print parameter names and shapes
        #for name, param in base_model_params.items():
        #    print(f"Parameter: {name}, Shape: {param.shape}")

        base_flattened_vector = torch.cat([param.to(self.device).view(-1) for param in base_model_params.values()])
        #print("Total number of parameters:", base_flattened_vector.numel())
        #("Shape of the flattened base model vector:", base_flattened_vector.shape)

        all_models = list(self.loaded_models.values())
        merged_model_dict = self.process_models(base_model_params, all_models)

        #print("Geometric median applied to the base model parameters.")
        self.merged_model = merged_model_dict  # Store the merged model
        assert len(self.merged_model) > 0, "Merged model is empty"

        self.base_model.load_state_dict(self.merged_model, strict=False)  # Load parameters directly from the merged model dict

        self.base_model.eval()  # Set to evaluation model
        return self.base_model

    def process_models(self, base_model_params: Dict[str, torch.Tensor], all_models: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        for param_name, param in base_model_params.items():  # Each dictionary contains parameter name and tensor

            # Skip certain parameters
            if any(substring in param_name for substring in [".q.", ".v."]):

                task_vector, original_shape = self.process_task_vectors(param_name, param, all_models)
                if task_vector:
                    # Compute geometric median after processing all models
                    result_for_block = self.compute_geometric_median(task_vector)
                    #print(result_for_block)  # Log for debugging

                    # Rescale the result back to the original shape and update the model
                    result_for_block_rescaled = result_for_block.view(original_shape)
                    #print("Rescaled Shape: ", result_for_block_rescaled.shape)

                    # Update merged_model with the rescaled result
                    base_model_params[param_name] += result_for_block_rescaled
            else:
                # If no task vectors were processed, retain the original parameter
                continue

        return base_model_params

    def process_task_vectors(self, param_name: str, param: torch.Tensor, all_models: List[Dict[str, torch.Tensor]]) -> Tuple[List[torch.Tensor], torch.Size]:
        task_vector = []
        original_shape = None  # Initialize original_shape

        for model in all_models:
            vector_A, vector_B = None, None
            
            # Initialize stir_q and stir_v before the loop to avoid UnboundLocalError
            stir_q = ""
            stir_v = ""
            
            # Iterate over each model's parameters
            for fine_tuned_param_name, fine_tuned_param in model.items():
                # Find ".q." and ".v." substring locations
                index_q = param_name.find(".q.")
                index_v = param_name.find(".v.")

                if index_q != -1:
                    stir_q = param_name[:index_q+3]  # Get the ".q." portion for matching
                    stir_v = ""
                if index_v != -1:
                    stir_v = param_name[:index_v+3]  # Get the ".v." portion for matching
                    stir_q = ""

                # Handle ".q." logic
                if ".q." in fine_tuned_param_name and stir_q in fine_tuned_param_name:
                    if "lora_A." in fine_tuned_param_name:
                        vector_A = fine_tuned_param
                    elif "lora_B." in fine_tuned_param_name:
                        vector_B = fine_tuned_param
                # Handle ".v." logic
                elif ".v." in fine_tuned_param_name and stir_v in fine_tuned_param_name:
                    if "lora_A." in fine_tuned_param_name:
                        vector_A = fine_tuned_param
                    elif "lora_B." in fine_tuned_param_name:
                        vector_B = fine_tuned_param

            # If both vectors A and B are found, multiply them and flatten the result
            if vector_A is not None and vector_B is not None:
                #print(vector_A.shape, vector_B.shape)
                result = torch.matmul(vector_B, vector_A)  # Element-wise multiplication
                original_shape = result.shape  # Save the original shape
                flattened_model_vector = result.view(-1)  # Flatten the result
                #print(original_shape)  # Log the original shape
                task_vector.append(flattened_model_vector)

        return task_vector, original_shape

    def compute_geometric_median(self, task_vectors: List[torch.Tensor], eps: float = 1e-8, max_iter: int = 300) -> torch.Tensor:
        if not task_vectors:
            raise ValueError("No task vectors provided for geometric median computation.")

        median = torch.mean(torch.stack(task_vectors), dim=0).to(self.device)

        for iteration in range(max_iter):
            distances = torch.stack([torch.norm(tv - median) for tv in task_vectors])
            distances[distances < 1e-10] = 1e-10  # Avoid division by zero
            weights = 1.0 / distances
            weights_sum = torch.sum(weights)

            if weights_sum == 0:
                #print("Weights sum to zero during geometric median computation.")
                break

            weighted_sum = torch.stack([w * tv for w, tv in zip(weights, task_vectors)]).sum(dim=0)
            new_median = weighted_sum / weights_sum
            shift = torch.norm(new_median - median)
            #print(f"Iteration {iteration}: shift={shift.item()}")

            if shift < eps:
                #print(f"Converged after {iteration} iterations.")
                return new_median

            median = new_median

        #print(f"Geometric median did not converge after {max_iter} iterations. Returning the last estimate.")
        return median


