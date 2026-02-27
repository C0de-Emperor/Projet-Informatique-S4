from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Mathematics import DiscreteFunction

def saltAndPaperNoising(discreteFunction: "DiscreteFunction", probability: float):
    import random
    
    for j in range(discreteFunction.height):
        for i in range(discreteFunction.width):
            if random.random() <= probability:
                discreteFunction[i, j] = 255 * random.choice([0, 1])
    

    
def randomNoising (discreteFunction: "DiscreteFunction", minAdd: int, maxAdd: int):
    import random

    for j in range(discreteFunction.height):
        for i in range(discreteFunction.width):
            value = discreteFunction[i, j] + random.randint(int(minAdd), int(maxAdd))

            discreteFunction[i, j] = value