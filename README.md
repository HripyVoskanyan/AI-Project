# Othello/Reversi

<p>Othello game between Human Player and AI Player</p>
<p>This project also includes a GUI.</p>

Three Game Modes
----------
* Easy
* Medium
* Hard

# AI Algorithm


H-Minimax Search
--------------
The H-Minimax Search takes heuristic functions to treat non-terminal states as terminal states. It also includes Alpha-Beta Pruning for space efficiency. 


Evaluation Function
-------------------
* Iago Move : Blocking Move and Corner Move. <br>
Blocking Move: Detects if the player can do a move to make the opponent have no valid moves. <br>
Corner Move: Detects if the player can do a move in the corner,

* Heuristic functions: If the eveluation function can't do Iago Move, it does h-minimax.

Heuristic Functions
-------------------
As heuristic functions are important part for the search algorithm, we have used different heuristic functions.

* Mobility (Computes the number of legal moves for the current and opponent players. Then it finds the difference between these numbers.)

* Stability (Computes the number of discs that are not valid to be outflanked till the game ends for the current and opponent players. Finds the difference between these numbers.)

* Pieces (Computes the difference of discs of current player and opponent player)

* Corner Move (Checks whether on next move the player can put a disc on the corner. Returns either 100 or 0.)

* Frontier Discs (Computes the difference of each square of the opponent on the board in eight different directions that are still empty. )

* Placement (Computes the difference between the scores according to the matrix of values for each square.)


        [[ 100, -10,  8,  6,  6,  8, -10, 100 ]        
        [ -10, -25, -4, -4, -4, -4, -25, -10 ]    
        [   8,  -4,  6,  4,  4,  6,  -4,   8 ] 
        [   6,  -4,  4,  0,  0,  4,  -4,   6 ]
    M = [   6,  -4,  4,  0,  0,  4,  -4,   6 ]
        [   8,  -4,  6,  4,  4,  6,  -4,   8 ]
        [ -10, -25, -4, -4, -4, -4, -25, -10 ]
        [ 100, -10,  8,  6,  6,  8, -10, 100 ]]

Necessary Libraries
-------------------
For running the code a user needs to have python 3.0. <br>
All the libraries are imported in main.py. <br>
If the user does not have them in the computer, they can install them, using pip install in the terminal. For example:
```python
pip install python-math
```

Steps to Run the Code
-------------------
* Download the zip
* Extract the files
* Run the main.py
* Choose the mode and play
* In order to choose a different heuristic write in the eval method of Evaluator class, the following:
```python
#for example for stability heuristic
score = stability_heuristic(array, player)
* ```


