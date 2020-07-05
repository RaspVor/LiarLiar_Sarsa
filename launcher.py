import random
import numpy as np
import numpy.random
import pandas as pd
from collections import defaultdict
import sys

def launcher_stochastic(Q, epsilon = 1.0/((80000/8000)+1)):
    
    
    # Class

    class nb_players:
        def __init__(self, nb=2, listing = np.array([])):
            self.nb=nb
            self.listing=listing
    
    class Player:
    
        def __init__(self, name = "Anonymous", player_number = int, cards = list(), cave = list()):
            self.name = name
            self.player_number = player_number
            self.cards = cards
            self.cave = cave
    
    class cards_game:
        def __init__(self, cards_list = np.array(["Ifrit","Ifrit","Ifrit","Ifrit","Ifrit","Ifrit","Ifrit","Ifrit",
                                                  "Shiva","Shiva","Shiva","Shiva","Shiva","Shiva","Shiva","Shiva",
                                                  "Ondine","Ondine","Ondine","Ondine","Ondine","Ondine","Ondine","Ondine",
                                                  "Ahuri","Ahuri","Ahuri","Ahuri","Ahuri","Ahuri","Ahuri","Ahuri",
                                                  "Bahamut","Bahamut","Bahamut","Bahamut","Bahamut","Bahamut","Bahamut","Bahamut",
                                                  "Leviathan","Leviathan","Leviathan","Leviathan","Leviathan","Leviathan","Leviathan","Leviathan",
                                                  "Golgotha","Golgotha","Golgotha","Golgotha","Golgotha","Golgotha","Golgotha","Golgotha",
                                                  "Taurus","Taurus","Taurus","Taurus","Taurus","Taurus","Taurus","Taurus"]),
                           last_cards=np.array([])
                     ):
            self.cards_list = cards_list
            self.last_cards = last_cards
    
    class dice:
        def __init__(self, winner = 0, players_list = []):
            self.winner = winner
            self.players_list = players_list
        
        def random_choose(self):
            self.winner = random.randint(1,len(self.players_list))-1
        
    
    
    class round_nb:
        def __init__(self, round_num = 1, player_start = 0, player_next = 0):  
            self.round_num = round_num
            self.player_start = player_start
            self.player_next = player_next
    
    
    #Mélanger le jeu de cartes
    def shuffle_deck(deck):
        deck_copy = deck
        np.random.shuffle(deck_copy)
        return(deck_copy)
    
    
    def one_turn_Bot_picker_Action(player):
        #exemple player =  Players_list[1]
        
        Picker_options = np.minimum([1,1,1,1,1,1,1,1], player.cards[1])
        temp = len(Picker_options[Picker_options>0])
        
        proba=[1/temp,1/temp,1/temp,1/temp,1/temp,1/temp,1/temp,1/temp]
        Picker_options = Picker_options*proba
    
        
        IA_pick_choice = np.random.choice(8, size=1, p=Picker_options)[0]
        
        Action_pickerpick_list = np.array([[1,0,0,0,0,0,0,0],
                                           [0,1,0,0,0,0,0,0],
                                           [0,0,1,0,0,0,0,0],
                                           [0,0,0,1,0,0,0,0],
                                           [0,0,0,0,1,0,0,0],
                                           [0,0,0,0,0,1,0,0],
                                           [0,0,0,0,0,0,1,0],
                                           [0,0,0,0,0,0,0,1]]
                                          )
        
        IA_pick_call = np.random.choice(2, size=1, p=[1/6,5/6])[0]  
        
        Action_pickercall_list = np.array([[1,0],[0,1]])
          
        return(Action_pickerpick_list[IA_pick_choice],
               Action_pickercall_list[IA_pick_call],
               np.append(Action_pickercall_list[IA_pick_call],Action_pickerpick_list[IA_pick_choice]))
    
    #xxx one_turn_Bot_picker_Action(Players_list[0].cards[1])
    #xxx> (array([0, 0, 1, 0, 0, 0, 0, 0]),
    #xxx> array([0, 1]),
    #xxx> array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0]))
           
           
    def one_turn_Bot_caller_Action():
        
        IA_call_call = np.random.choice(2, size=1, p=[1/6,5/6])[0]  
        
        Action_callercall_list = np.array([[1,0],[0,1]])
        
        return(Action_callercall_list[IA_call_call])
    
    #xxx one_turn_Bot_caller_Action()
    #xxx> array([1, 0])
    
    
    #Control des probas
    def get_probs_pick_actions(Q_s, epsilon, nS = 16, nA=18):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(nA) * epsilon / nS
        best_a = np.argmax(Q_s[:-2])
        policy_s[best_a] = 1 - epsilon + (epsilon / nS)
        
        selector = np.array([1,1,1,1,
                             1,1,1,1,
                             1,1,1,1,
                             1,1,1,1,
                             0,0])
        policy_s = np.where(selector==0, 0, policy_s)
        
        return policy_s


    def get_probs_call_actions(Q_s, epsilon, nS = 2, nA=18):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(nA) * epsilon / nS
        best_a = np.argmax(Q_s[-2:])+16
        policy_s[best_a] = 1 - epsilon + (epsilon / nS)
        
        selector = np.array([0,0,0,0,
                             0,0,0,0,
                             0,0,0,0,
                             0,0,0,0,
                             1,1])
        policy_s = np.where(selector==0, 0, policy_s)
        
        return policy_s
    
    
    
    # Initialisation du jeu

    ##Nombre de joueurs
    Nb_players = nb_players(nb=2, listing = np.arange(0,2))
    
    #xxx Nb_players.nb
    #xxx> 2
    
    ##Definition des joueurs
    
    Player1 = Player(name = "Joueur_1", player_number = 0)
    Player2 = Player(name = "Joueur_2", player_number = 1)
    Players_list = [Player1,Player2]
    
    #xxx Player.players_list[0].name
    #xxx> 'Joueur_1'
    
    ##Mélange des cartes (fonctionne si plus de 2 joueurs)
    Deck = cards_game()
    
    #xxx Deck.cards_list
    if Nb_players.nb == 2:
        Deck.cards_list = shuffle_deck(Deck.cards_list)[:-10] 
    else:
        Deck.cards_list = shuffle_deck(Deck.cards_list)
    
    if len(Deck.cards_list) % Nb_players.nb != 0:
        Deck.last_cards = Deck.cards_list[-1]
        Deck.cards_list = Deck.cards_list[:-1]
        
    
    ##Distribution des cartes (fonctionne si plus de 2 joueurs)
    for i in Players_list:
        df1 = pd.DataFrame({'key': ['Ahuri', 'Bahamut', 'Golgotha', 'Ifrit', 'Leviathan', 'Ondine','Shiva', 'Taurus']})
        
        if i.player_number < Nb_players.nb:
            tempx = np.unique(list(np.sort(Deck.cards_list[:len(Deck.cards_list)//Nb_players.nb*Nb_players.nb].reshape(len(Deck.cards_list)//Nb_players.nb,Nb_players.nb)[:,i.player_number])), return_counts=True)
            df2 = pd.DataFrame({'key': tempx[0],
                                'value': tempx[1]})
            df = df1.merge(df2, on='key', how='left').fillna(0)
            df['value'] = df['value'].apply(np.int64)
            
            i.cards = (np.array(df['key'], dtype='<U9'), np.array(df['value'], dtype='int64'))
            i.cave = (np.array(['Ahuri', 'Bahamut', 'Golgotha', 'Ifrit', 'Leviathan', 'Ondine', 
                                'Shiva', 'Taurus'], dtype='<U9'), np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype='int64'))
        else:
            tempx = np.unique(list(np.sort(np.append(Deck.cards_list[:len(Deck.cards_list)//Nb_players.nb*Nb_players.nb].reshape(len(Deck.cards_list)//Nb_players.nb,Nb_players.nb)[:,i.player_number],Deck.last_cards))), return_counts=True)
            
            df2 = pd.DataFrame({'key': tempx[0],
                                'value': tempx[1]})
            df = df1.merge(df2, on='key', how='left').fillna(0)
            df['value'] = df['value'].apply(np.int64)
            
            i.cards = (np.array(df['key'], dtype='<U9'), np.array(df['value'], dtype='int64'))
            i.cave = (np.array(['Ahuri', 'Bahamut', 'Golgotha', 'Ifrit', 'Leviathan', 'Ondine', 
                                'Shiva', 'Taurus'], dtype='<U9'), np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype='int64'))
        
            
    ## Qui commence
    ##On lance le dé
    
    de = dice(0,Players_list)
    de.random_choose()
    
    #xxx de.winner
    #xxx> 0 ou 1 si 2 joueurs
    
    #Initialize Counter
    start_ind = 1
    compteur_tour = round_nb(start_ind, Nb_players.listing[de.winner], (Nb_players.listing[de.winner]+1)%len(Nb_players.listing))
    
    #xxx compteur_tour.player_start
    #xxx> 1
    
    #Initialize Game status
    game_end = False
    
    #Initialize Reward
    Reward = 0
    
    #Initialize Actions
    RL_actions = np.array([[1,0,1,0,0,0,0,0,0,0],
                           [1,0,0,1,0,0,0,0,0,0],
                           [1,0,0,0,1,0,0,0,0,0],
                           [1,0,0,0,0,1,0,0,0,0],
                           [1,0,0,0,0,0,1,0,0,0],
                           [1,0,0,0,0,0,0,1,0,0],
                           [1,0,0,0,0,0,0,0,1,0],
                           [1,0,0,0,0,0,0,0,0,1],
                           [0,1,1,0,0,0,0,0,0,0],
                           [0,1,0,1,0,0,0,0,0,0],
                           [0,1,0,0,1,0,0,0,0,0],
                           [0,1,0,0,0,1,0,0,0,0],
                           [0,1,0,0,0,0,1,0,0,0],
                           [0,1,0,0,0,0,0,1,0,0],
                           [0,1,0,0,0,0,0,0,1,0],
                           [0,1,0,0,0,0,0,0,0,1],
                           [1,0,0,0,0,0,0,0,0,0],
                           [0,1,0,0,0,0,0,0,0,0]
                           ]) 
    
    
    Zactions =	{
                  "[1 0 1 0 0 0 0 0 0 0]" : 0, 
                  "[1 0 0 1 0 0 0 0 0 0]" : 1, 
                  "[1 0 0 0 1 0 0 0 0 0]" : 2, 
                  "[1 0 0 0 0 1 0 0 0 0]" : 3, 
                  "[1 0 0 0 0 0 1 0 0 0]" : 4, 
                  "[1 0 0 0 0 0 0 1 0 0]" : 5, 
                  "[1 0 0 0 0 0 0 0 1 0]" : 6, 
                  "[1 0 0 0 0 0 0 0 0 1]" : 7, 
                  "[0 1 1 0 0 0 0 0 0 0]" : 8, 
                  "[0 1 0 1 0 0 0 0 0 0]" : 9, 
                  "[0 1 0 0 1 0 0 0 0 0]" : 10, 
                  "[0 1 0 0 0 1 0 0 0 0]" : 11, 
                  "[0 1 0 0 0 0 1 0 0 0]" : 12, 
                  "[0 1 0 0 0 0 0 1 0 0]" : 13, 
                  "[0 1 0 0 0 0 0 0 1 0]" : 14, 
                  "[0 1 0 0 0 0 0 0 0 1]" : 15, 
                  "[1 0 0 0 0 0 0 0 0 0]" : 16, 
                  "[0 1 0 0 0 0 0 0 0 0]" : 17
                }
    
    RL_pick_actions_proba = np.array([1/16,1/16,1/16,1/16,
                                      1/16,1/16,1/16,1/16,
                                      1/16,1/16,1/16,1/16,
                                      1/16,1/16,1/16,1/16,
                                      0,0])
    
    
    
    
    RL_call_actions_proba = np.array([0,0,0,0,
                                      0,0,0,0,
                                      0,0,0,0,
                                      0,0,0,0,
                                      1/2,1/2]) 
    
    #Initialisation Action
    prob =  np.zeros(18)
         
    action =  np.zeros(18)
    
    #Initialise Episode 
    episode = []
    
    
    # Deroulement du jeu
    
    while (game_end == False):
    
        #print('Reward :',Reward)
        #print('compteur_tour.round_num :',compteur_tour.round_num)
        #print('compteur_tour.player_start :',compteur_tour.player_start)
        #print('Players_list[0].cards[1] :',Players_list[0].cards[1])
        #print('Players_list[0].cave[1] :',Players_list[0].cave[1])
        #print('Players_list[1].cards[1] :',Players_list[1].cards[1])
        #print('Players_list[1].cave[1] :',Players_list[1].cave[1])
        
        State = (str(compteur_tour.player_start)+str(Players_list[0].cards[1])+str(Players_list[0].cave[1])+str(Players_list[1].cave[1]))
       
         
        if compteur_tour.player_start != 0:
            
            #Initialisation Action/prob
            prob = get_probs_call_actions(Q[State], epsilon) \
                                    if (((State in Q) == True) and (np.all(Q[State][-2:] ==0) == False)) else RL_call_actions_proba
                                      
            #print("\rMAX : {} // Q[State]: {}.".format(np.argmax(Q[State][-2:])+16  , Q[State]), end="")          
            #sys.stdout.flush()
            
            #IA pick
            turn_played_IA = one_turn_Bot_picker_Action(Players_list[compteur_tour.player_start])
            
            #The card picked is removed from his deck
            temp =  (np.array(['Ahuri', 'Bahamut', 'Golgotha', 'Ifrit', 'Leviathan', 'Ondine','Shiva', 'Taurus'], dtype='<U9'), 
                     Players_list[compteur_tour.player_start].cards[1] - turn_played_IA[0])
            Players_list[compteur_tour.player_start].cards = temp
            
            #RL make a call
            RL_call_choice = np.random.choice(18, size=1, p=prob)[0]
            RL_call_action = RL_actions[RL_call_choice][:2]
            Action = RL_actions[RL_call_choice] #save the action
            
            #We check who won
            if ((RL_call_action == turn_played_IA[1]).all() == True):
                temp =  (np.array(['Ahuri', 'Bahamut', 'Golgotha', 'Ifrit', 'Leviathan', 'Ondine','Shiva', 'Taurus'], dtype='<U9'), 
                         Players_list[compteur_tour.player_start].cave[1] + turn_played_IA[0])
                Players_list[compteur_tour.player_start].cave = temp
                compteur_tour = round_nb(compteur_tour.round_num +1 , compteur_tour.player_start, (compteur_tour.player_start+1)%len(Nb_players.listing))
            else:
                temp =  (np.array(['Ahuri', 'Bahamut', 'Golgotha', 'Ifrit', 'Leviathan', 'Ondine','Shiva', 'Taurus'], dtype='<U9'), 
                         Players_list[compteur_tour.player_next].cave[1] + turn_played_IA[0])
                Players_list[compteur_tour.player_next].cave = temp
                compteur_tour = round_nb(compteur_tour.round_num +1 , compteur_tour.player_next, (compteur_tour.player_next+1)%len(Nb_players.listing))
            
            
            #Reward
            if (len(Players_list[compteur_tour.player_start].cards[1][Players_list[compteur_tour.player_start].cards[1] < 0]) > 0):
                game_end = True
                Reward = 10000
            elif (len(Players_list[compteur_tour.player_start].cave[1][Players_list[compteur_tour.player_start].cave[1] > 3]) > 0):
                game_end = True
                Reward = 10000
            elif (len(Players_list[compteur_tour.player_next].cave[1][Players_list[compteur_tour.player_next].cave[1] > 3]) > 0):
                game_end = True
                Reward = -10000
            elif ((RL_call_action == turn_played_IA[1]).all() == True):
                Reward = 10
            else:
                Reward = -10
            
            episode.append((State, str(Action), Reward))
            
        else:
            
            #Initialisation Action/prob
            prob = get_probs_pick_actions(Q[State], epsilon) \
                                    if(((State in Q) == True) and (np.all(Q[State][:-2] ==0.) == False)) else RL_pick_actions_proba
                                        
            #print("\rMAX : {} // Q[State]: {}.".format(np.argmax(Q[State][:-2])  , Q[State]), end="") 
            #sys.stdout.flush()
            
            #RL pick
            RL_pick_choice = np.random.choice(18, size=1, p=prob)[0]
            RL_pick_action = RL_actions[RL_pick_choice]
            
            #Card pick and call made
            RL_pickcard_action = RL_pick_action[2:10]
            RL_pickcall_action = RL_pick_action[:2]
            Action = RL_actions[RL_pick_choice] #save the action
            
            #The card picked is removed from his deck
            temp =  (np.array(['Ahuri', 'Bahamut', 'Golgotha', 'Ifrit', 'Leviathan', 'Ondine','Shiva', 'Taurus'], dtype='<U9'), 
                     Players_list[compteur_tour.player_start].cards[1] - RL_pickcard_action)
            Players_list[compteur_tour.player_start].cards = temp
            
            #IA make a call
            IA_call_choice = one_turn_Bot_caller_Action()
            
            
            #We check who won
            if ((RL_pickcall_action ==  IA_call_choice).all() == True):
                temp = (np.array(['Ahuri', 'Bahamut', 'Golgotha', 'Ifrit', 'Leviathan', 'Ondine','Shiva', 'Taurus'], dtype='<U9'), 
                        Players_list[compteur_tour.player_start].cave[1] + RL_pickcard_action)
                Players_list[compteur_tour.player_start].cave = temp
                compteur_tour = round_nb(compteur_tour.round_num +1 , compteur_tour.player_start, (compteur_tour.player_start+1)%len(Nb_players.listing))
            else:
                temp = (np.array(['Ahuri', 'Bahamut', 'Golgotha', 'Ifrit', 'Leviathan', 'Ondine','Shiva', 'Taurus'], dtype='<U9'), 
                        Players_list[compteur_tour.player_next].cave[1] + RL_pickcard_action)
                Players_list[compteur_tour.player_next].cave = temp
                compteur_tour = round_nb(compteur_tour.round_num +1 , compteur_tour.player_next, (compteur_tour.player_next+1)%len(Nb_players.listing))
            
            #Reward
            if (len(Players_list[compteur_tour.player_start].cards[1][Players_list[compteur_tour.player_start].cards[1] < 0]) > 0):
                game_end = True
                Reward = -10000
            elif (len(Players_list[compteur_tour.player_start].cave[1][Players_list[compteur_tour.player_start].cave[1] > 3]) > 0):
                game_end = True
                Reward = -10000
            elif (len(Players_list[compteur_tour.player_next].cave[1][Players_list[compteur_tour.player_next].cave[1] > 3]) > 0):
                game_end = True
                Reward = 10000
            elif ((RL_pickcall_action ==  IA_call_choice).all() == True):
                Reward = -10
            else:
                Reward = 10
                
            episode.append((State, str(Action), Reward))
            
    
    return episode

Q = defaultdict(lambda: np.zeros(18))
launcher_stochastic(Q)
