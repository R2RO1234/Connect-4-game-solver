import generate_boards
import os
import numpy as np


#  needs to have a method to say where everything is saved
class BoardDataset:
    def __init__(self, name: str, output_root="output"):
        """
        Initializes a board dataset with a specific name.
        Creates a directory structure:
        output_root/
            name/
                dict_output/
                evaluation_scores/
        """
        self.name = name
        self.root_dir = os.path.join(output_root, name)
        self.dict_dir = os.path.join(self.root_dir, "dict_output")
        self.evaluation_dir = os.path.join(self.root_dir, "evaluation")
        self.evaluated_boards_dir = os.path.join(self.evaluation_dir, "boards")
        self.evaluated_metadata_dir = os.path.join(self.evaluation_dir, "metadata")
        os.makedirs(output_root, exist_ok=True)
        os.makedirs(self.dict_dir, exist_ok=True)
        os.makedirs(self.evaluation_dir, exist_ok=True)
        os.makedirs(self.evaluated_boards_dir, exist_ok=True)
        os.makedirs(self.evaluated_metadata_dir, exist_ok=True)

        self.boards_file = os.path.join(self.dict_dir, "boards.npy")
        self.hashes_file = os.path.join(self.dict_dir, "hashes.txt")
        
        self.evaluated_boards = self.load_evaluated_boards()
        self.boards_dict = self.load_dict()

    def load_dict(self)-> dict:
        self.boards_dict = generate_boards.get_dict_from_files(self.hashes_file , self.boards_file)
        return self.boards_dict

    def save_boards(self):
        generate_boards.write_boards(self.get_values_dict() ,self.boards_file )
        generate_boards.write_hashes(self.get_keys_dict() ,self.hashes_file)
    
    def expand_dict_and_save(self, num_games):
        self.boards_dict = generate_boards.generate_board_states(self.boards_dict , num_games)
        self.save_boards()


    def get_values_dict(self):
        return list(self.boards_dict.values())
    def get_keys_dict(self):
        return list(self.boards_dict.keys())
    
    
    
    def evaluate_remaining_boards(self, agent, num_batches):
        already_evaluated = len(self.evaluated_boards)
        generate_boards.evaluate_boards_in_batches(agent,self.get_values_dict() , num_batches , 
                                                   self.evaluated_boards_dir , self.evaluated_metadata_dir , already_evaluated)

    def get_num_non_evaluated_boards(self):
        self.load_dict()
        self.load_evaluated_boards()
        return len(self.boards_dict) - len(self.evaluated_boards)
    
    def load_evaluated_boards(self) -> np.ndarray:
        """
        Loads all .npy files of the dataset into a single NumPy array.
        Raises:
            ValueError: If any file in the directory is not a .npy file.
        """
        all_boards = []
        
        for file_name in sorted(os.listdir(self.evaluated_boards_dir)):
            file_path = os.path.join(self.evaluated_boards_dir, file_name)
            if os.path.isfile(file_path):
                if not file_name.endswith(".npy"):
                    raise ValueError(f"Unexpected file type found: {file_name}")
                all_boards.append(np.load(file_path))
        
        if not all_boards:
            return np.array([])  # empty directory
        self.evaluated_boards =  np.concatenate(all_boards, axis=0)
        return self.evaluated_boards