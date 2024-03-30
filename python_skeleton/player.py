"""
Simple example pokerbot, written in Python.
"""

import random
import pickle
import itertools
from typing import Optional
from handCombinations import generate_and_categorize_hands
from skeleton.actions import Action, CallAction, CheckAction, FoldAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

class Player(Bot):
    """
    A pokerbot.
    """

    def __init__(self) -> None:
        """
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        """
        self.log = []
        self.pre_computed_probs = pickle.load(open("python_skeleton/skeleton/pre_computed_probs.pkl", "rb"))
        pass

    def handle_new_round(self, game_state: GameState, round_state: RoundState, active: int) -> None:
        """
        Called when a new round starts. Called NUM_ROUNDS times.
        
        Args:
            game_state (GameState): The state of the game.
            round_state (RoundState): The state of the round.
            active (int): Your player's index.

        Returns:
            None
        """
        #my_bankroll = game_state.bankroll # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        #game_clock = game_state.game_clock # the total number of seconds your bot has left to play this game
        #round_num = game_state.round_num # the round number from 1 to NUM_ROUNDS
        #my_cards = round_state.hands[0] # your cards
        #big_blind = bool(active) # True if you are the big blind
        self.log = []
        self.log.append("================================")
        self.log.append("new round")
        pass

    def handle_round_over(self, game_state: GameState, terminal_state: TerminalState, active: int, is_match_over: bool) -> Optional[str]:
        """
        Called when a round ends. Called NUM_ROUNDS times.

        Args:
            game_state (GameState): The state of the game.
            terminal_state (TerminalState): The state of the round when it ended.
            active (int): Your player's index.

        Returns:
            Your logs.
        """
        #my_delta = terminal_state.deltas[active] # your bankroll change from this round
        #previous_state = terminal_state.previous_state # RoundState before payoffs
        #street = previous_state.street # 0, 3, 4, or 5 representing when this round ended
        #my_cards = previous_state.hands[0] # your cards
        #opp_cards = previous_state.hands[1] # opponent's cards or [] if not revealed
        self.log.append("game over")
        self.log.append("================================\n")

        return self.log

    def estimate_opponent_equity(self, my_cards, board_cards, end):
        """
        Estimate the opponent hand probabilities based on the cards on the board and your own cards.
        """
        # Generate all possible opponent hands
        possible_opponent_hands = []
        for card in board_cards:
            possible_opponent_hands.append([card + 'h', card + 'd'])
            possible_opponent_hands.append([card + 'h', card + 's'])
            possible_opponent_hands.append([card + 'd', card + 's'])

        # Remove duplicates
        possible_opponent_hands = list(set([tuple(sorted(hand)) for hand in possible_opponent_hands]))

        # Remove cards that are already on the board
        for card in board_cards:
            possible_opponent_hands = [hand for hand in possible_opponent_hands if card not in hand]

        # Remove cards that are already in your hand
        for card in my_cards:
            possible_opponent_hands = [hand for hand in possible_opponent_hands if card not in hand]
        if end:
            return possible_opponent_hands
        # Calculate the probability of each possible opponent hand
        opponent_hand_probs = {}
        for hand in possible_opponent_hands:
            opponent_hand_probs['_'.join(hand)] = self.pre_computed_probs['_'.join(hand) + '_' + '_'.join(sorted(board_cards))]

        return opponent_hand_probs

    #this code just returns if their hand has more equity than mine
    def hand_rank(self, observation, opp_hand):
        my_equity = self.pre_computed_probs['_'.join(sorted(observation["my_cards"])) + '_' + '_'.join(sorted(observation["board_cards"]))]
        opp_equity = self.pre_computed_probs['_'.join(sorted(opp_hand)) + '_' + '_'.join(sorted(observation["board_cards"]))]
        return opp_equity > my_equity



    def get_action(self, observation: dict) -> Action:
        """
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Args:
            observation (dict): The observation of the current state.
            {
                "legal_actions": List of the Actions that are legal to take.
                "street": 0, 1, or 2 representing pre-flop, flop, or river respectively
                "my_cards": List[str] of your cards, e.g. ["1s", "2h"]
                "board_cards": List[str] of the cards on the board
                "my_pip": int, the number of chips you have contributed to the pot this round of betting
                "opp_pip": int, the number of chips your opponent has contributed to the pot this round of betting
                "my_stack": int, the number of chips you have remaining
                "opp_stack": int, the number of chips your opponent has remaining
                "my_bankroll": int, the number of chips you have won or lost from the beginning of the game to the start of this round
                "min_raise": int, the smallest number of chips for a legal bet/raise
                "max_raise": int, the largest number of chips for a legal bet/raise
            }

        Returns:
            Action: The action you want to take.
        """
        my_contribution = STARTING_STACK - observation["my_stack"] # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - observation["opp_stack"] # the number of chips your opponent has contributed to the pot
        continue_cost = observation["opp_pip"] - observation["my_pip"] # the number of chips needed to stay in the pot

        self.log.append("My cards: " + str(observation["my_cards"]))
        self.log.append("Board cards: " + str(observation["board_cards"]))
        self.log.append("My stack: " + str(observation["my_stack"]))
        self.log.append("My contribution: " + str(my_contribution))
        self.log.append("My bankroll: " + str(observation["my_bankroll"]))

        ###replace this code: this is just a rule based agent that plays any pair, same suited hand, or hands with straight potential

        #write code to estimate the opponent hand probabilities
        #sum up all the probabilities of the opponent having a hand better than yours
 
        #my_equity = self.pre_computed_probs['_'.join(sorted(observation["my_cards"])) + '_' + '_'.join(sorted(observation["board_cards"]))]

        # Estimate opponent hand probabilities
        street = observation["street"]
        if street < 2:
            opponent_equity = self.estimate_opponent_equity(observation["my_cards"], observation["board_cards"], False)
            # Sum up probabilities of opponent having a better hand
            sum_better_hands = sum([self.hand_rank(observation, hand) for hand, prob in opponent_equity.items()])/17550
            # Make decision based on the sum of probabilities
            if sum_better_hands > 0.5:
                return FoldAction()
            #sum_better_hands in between 0.4 and 0.5
            elif sum_better_hands > 0.4 and sum_better_hands <= 0.5:
                return RaiseAction(observation["min_raise"])
            elif sum_better_hands > 0.3 and sum_better_hands <= 0.4:
                amt = random.randint(observation["min_raise"], observation["max_raise"])
                return RaiseAction(amt)
            elif sum_better_hands <= 0.05:
                return RaiseAction(observation["max_raise"])
            
            if CallAction in observation["legal_actions"]:
                return CallAction()
            else:
                return CheckAction()
        else:
            my_hand = generate_and_categorize_hands(observation["my_cards"], observation["board_cards"])
            opp_hands = self.estimate_opponent_equity(observation["my_cards"], observation["board_cards"], True)
            results = [generate_and_categorize_hands(hand, observation["board_cards"]) for hand in opp_hands]
            win_prob = sum([my_hand < hand for hand in results])/len(results)
            if win_prob > 0.8:
                return RaiseAction(observation["max_raise"])
            elif win_prob > 0.5 and win_prob <= 0.8:
                amt = random.randint(observation["min_raise"], observation["max_raise"])
                return RaiseAction(amt)
            elif win_prob > 0.2 and win_prob <= 0.5:
                return RaiseAction(observation["min_raise"])
            else:
                if CheckAction in observation["legal_actions"]:
                    return CheckAction()
                else:
                    return FoldAction()



if __name__ == '__main__':
    run_bot(Player(), parse_args())