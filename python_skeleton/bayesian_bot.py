"""
Simple example pokerbot, written in Python.
"""

import random
import pickle
import itertools
from typing import Optional
import numpy as np

from skeleton.actions import Action, CallAction, CheckAction, FoldAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from skeleton.evaluate import evaluate

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
        #self.pre_computed_probs = pickle.load(open("python_skeleton/skeleton/pre_computed_probs.pkl", "rb"))
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

    def generate_prior_ours(self, observation):
        """
        Generate a prior from the pre_computed_probs pickle file.

        Args:
            observation (dict): The observation of the current state.

        Returns:
            prior (dict): the prior probability distribution.
        """
        my_cards = observation["my_cards"]
        board_cards = observation["board_cards"]
        prior = {}
        combo = observation["street"] + 2
        deck = [str(i) + suit for i in range(1, 10) for suit in ['s', 'h', 'd']]
        future_cards = [card for card in deck if card not in my_cards + board_cards]
        for hand in itertools.combinations(my_cards + board_cards + future_cards, combo):
            hand_str = "_".join(sorted(hand))
            prior[hand_str] = evaluate(my_cards, board_cards)
        return prior
    
    def generate_prior_theirs(self, observation):
        """
        Generate a prior from the pre_computed_probs pickle file.

        Args:
            observation (dict): The observation of the current state.

        Returns:
            prior (dict): the prior probability distribution.
        """
        my_cards = observation["my_cards"]
        board_cards = observation["board_cards"]
        prior = {}
        combo = observation["street"] + 2
        deck = [str(i) + suit for i in range(1, 10) for suit in ['s', 'h', 'd']]
        enemy_cards = [card for card in deck if card not in my_cards + board_cards]
        for hand in itertools.combinations(enemy_cards + board_cards, combo):
            hand_str = "_".join(sorted(hand))
            enemy = [card for card in hand if card not in board_cards]
            prior[hand_str] = evaluate(enemy, board_cards)
        return prior
    
    def mcmc_simulation(self, prior, num_samples):
        """
        Perform MCMC simulation to generate a posterior distribution of equity in a hand.

        Args:
            prior (dict): The prior probability distribution.
            num_samples (int): The number of MCMC samples to generate.

        Returns:
            posterior (dict): The posterior probability distribution.
        """
        posterior = {}
        ps = list(prior.values())/np.sum(list(prior.values()))
        samples = np.random.choice(list(prior.keys()), size=num_samples, p=ps)
        for hand in samples:
            if hand in posterior:
                posterior[hand] += prior[hand]
            else:
                posterior[hand] = 1
        for hand in posterior:
            posterior[hand] /= num_samples
        return posterior

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

        ###replace this code: this is just a rule based agent that plays any pair, same suited hand, or hands with straight potential

        #this gives us a prior distribution of our equity at each street
        prior_us = self.generate_prior_ours(observation)
        #this gives us a prior distribution of the opponent's equity at each street
        prior_enemy = self.generate_prior_theirs(observation)


        num_samples = 10000

        #this gives us a simulated posterior distribution of our equity
        posterior_us = self.mcmc_simulation(prior_us, num_samples)

        #this gives us a simulated posterior distribution of the opponent's equity
        posterior_enemy = self.mcmc_simulation(prior_enemy, num_samples)
        
        #prior update step based on first action
        #if this is true we have second action and need to update prior
        #based on opponent bet
        if CallAction in observation["legal_actions"]:
            opp_bet = observation["opp_pip"]
            if opp_bet == 400:
                epsilon = np.random.uniform(0, 1)
                if epsilon < 0.05:
                    return CallAction()
            if opp_bet > 200:
                my_equity = random.choice(list(posterior_us.values())[:len(posterior_us)//2])
                opp_equity = random.choice(list(posterior_enemy.values())[len(posterior_enemy)//2:])
            else:
                my_equity = random.choice(list(posterior_us.values())[len(posterior_us)//2:])
                opp_equity = random.choice(list(posterior_enemy.values())[:len(posterior_enemy)//2])
        
        #else pull random value from posteriors if we are first action
        else:
            opp_stack = observation["opp_stack"]
            if opp_stack == 400:
                my_equity = random.choice(list(posterior_us.values()))
                opp_equity = random.choice(list(posterior_enemy.values()))
            elif opp_stack > 200:
                my_equity = random.choice(list(posterior_us.values())[len(posterior_us)//2:])
                opp_equity = random.choice(list(posterior_enemy.values())[:len(posterior_enemy)//2])
            else:
                my_equity = random.choice(list(posterior_us.values())[:len(posterior_us)//2])
                opp_equity = random.choice(list(posterior_enemy.values())[len(posterior_enemy)//2:])

        #introduce irrationality in our agent
        epsilon = np.random.uniform(0, 1)
        if epsilon < 0.05:
            if RaiseAction in observation["legal_actions"]:
                amt = observation["max_raise"]
                return RaiseAction(amt)
            else:
                return FoldAction()
        
        self.log.append(f"Our equity: {my_equity}")
        self.log.append(f"Opponent equity: {opp_equity}")
        print("VA Linux")
        #otherwise, compare equities and proceed.
        equity_diff = ((my_equity - opp_equity)/opp_equity)
        if equity_diff > 0.2:
            if RaiseAction in observation["legal_actions"]:
                stack_bet = int(equity_diff * (observation["my_stack"]))
                amt = max(observation["min_raise"], stack_bet)
                return RaiseAction(amt)
            else:
                return CallAction()
        
        if CallAction in observation["legal_actions"]:
            return CallAction()
        elif CheckAction in observation["legal_actions"]:
            return CheckAction()
        else:
            return FoldAction()

if __name__ == '__main__':
    run_bot(Player(), parse_args())