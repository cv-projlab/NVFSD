from enum import Enum, auto
from typing import List, Union

class Task(Enum):
    A = 1
    B = 2
    C = 3
    D = 4
    E = 5


Task_A = Task.A.value
Task_B = Task.B.value
Task_C = Task.C.value
Task_D = Task.D.value
Task_E = Task.E.value

class TaskRecordingCombinations(Enum):
    """ Tasks-Recordings Combination 
    
    NOTE: Combination in form of dictionary such as 
    
    {
        't':[Task_A, ..., Task_Z], 
        'r':[[Recording_1_for_task_A, ..., Recording_99_for_task_A], 
                ..., 
            [Recording_1_for_task_Z, ..., Recording_99_for_task_Z]]
    }
    
    """

    A1 = {'t':[Task_A], 'r':[[1]]}
    A2 = {'t':[Task_A], 'r':[[2]]}
    A = {'t':[Task_A], 'r':[[1,2]]}
    B1 = {'t':[Task_B], 'r':[[1]]}
    B2 = {'t':[Task_B], 'r':[[2]]}
    B = {'t':[Task_B], 'r':[[1,2]]}
    C123 = {'t':[Task_C], 'r':[[1,2,3]]}
    C45 = {'t':[Task_C], 'r':[[4,5]]}
    C = {'t':[Task_C], 'r':[[1,2,3,4,5]]}
    E = {'t':[Task_E], 'r':[[1,2,3,4,5,6,7,8]]}

    A1B1 = {'t':[Task_A, Task_B], 'r':[[1], [1]]}
    A2B2 = {'t':[Task_A, Task_B], 'r':[[2], [2]]}
    AB = {'t':[Task_A, Task_B], 'r':[[1,2], [1,2]]}
    ABC123 = {'t':[Task_A, Task_B, Task_C], 'r':[[1,2], [1,2], [1,2,3]]}
    ABC = {'t':[Task_A, Task_B, Task_C], 'r':[[1,2], [1,2], [1,2,3,4,5]]}
    A1B1C123 = {'t':[Task_A, Task_B, Task_C], 'r':[[1], [1], [1,2,3]]}
    A2B2C45 = {'t':[Task_A, Task_B, Task_C], 'r':[[2], [2], [4,5]]}
    
    @staticmethod
    def assert_combination_exists(combination: str) -> None:
        assert combination in TaskRecordingCombinations._member_names_, f'Combination < {combination} > does not exist!'

    @staticmethod
    def get_available_combinations() -> List[str]:
        return TaskRecordingCombinations._member_names_