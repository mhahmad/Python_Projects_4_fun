import random
from enum import Enum

# Define an enumeration for the choices
class Choice(Enum):
    ROCK = 'rock'
    PAPER = 'paper'
    SCISSORS = 'scissors'

# Function to get a valid user choice
def get_user_choice():
    while True:
        user_input = input("Choose your hand: rock, paper, scissors: ").lower()
        # Check if the input is a valid choice
        if user_input in [choice.value for choice in Choice]:
            return Choice(user_input)
        else:
            print("Input is not valid. Please try again.")

# Function to determine the winner
def determine_winner(user_choice, pc_choice):
    if user_choice == pc_choice:
        return "It's a Tie!"
    elif (user_choice, pc_choice) in WINNING_CONDITIONS:
        return "You Win!"
    else:
        return "You Lose!"

# Define winning conditions
WINNING_CONDITIONS = {
    (Choice.ROCK, Choice.SCISSORS),
    (Choice.PAPER, Choice.ROCK),
    (Choice.SCISSORS, Choice.PAPER),
}

# Main game loop
def main():
    while True:
        user_choice = get_user_choice()
        pc_choice = random.choice(list(Choice))
        print("You chose:", user_choice.value)
        print("Other player chose:", pc_choice.value)
        print(determine_winner(user_choice, pc_choice))

        # Ask if the user wants to play again
        play_again = input("Do you want to play again? (yes/no): ").lower()
        if play_again != 'yes':
            break

# Entry point of the program
if __name__ == "__main__":
    main()
