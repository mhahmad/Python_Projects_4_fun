
# this is project one in project course series 
import random

R = 'rock'
P = 'paper'
S = 'scissors'
option_list = [R,P,S]

user_choice = input("choose you hand : rock , paper , scissors : ")
pc_choice = random.choice(option_list)

# validate input : 
if user_choice not in option_list :
    print("input is not valid !")
  
else:
    print("you chose : " , user_choice)
    print("other player chose : ",pc_choice)

    # user wins 
    if  (pc_choice== R and user_choice==P) or (pc_choice== S and user_choice==R) or (pc_choice== P and user_choice==S) :
        print("you Win ! ")
    #user loses
    elif  (pc_choice== P and user_choice==R) or (pc_choice== R and user_choice==S) or (pc_choice== S and user_choice==P) :
        print("you Lose ! ")
    #tie
    else :
        print("its a Tie :- ")
