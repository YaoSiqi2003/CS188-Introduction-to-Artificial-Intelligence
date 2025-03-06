# factorOperations.py
# -------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from typing import List
from bayesNet import Factor
import functools
from util import raiseNotDefined

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors: List[Factor], joinVariable: str):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        vars_on_left = [factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()]
        if numVariableOnLeft > 1:
            print("Factor failed joinFactorsByVariable typecheck: ", vars_on_left)
            raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
        
        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()

########### ########### ###########
########### QUESTION 2  ###########
########### ########### ###########

def joinFactors(factors: List[Factor]):
    # Ensure factors is a list in case dict_values was passed in
    factors = list(factors)

    # Step 1: Gather unconditioned and conditioned variables
    unconditioned_vars = set()
    conditioned_vars = set()
    for factor in factors:
        unconditioned_vars.update(factor.unconditionedVariables())
        conditioned_vars.update(factor.conditionedVariables())

    # Remove variables that are in unconditioned_vars from conditioned_vars
    conditioned_vars -= unconditioned_vars

    # Step 2: Get the variableDomainsDict from any one of the factors (they are all the same)
    variable_domains_dict = factors[0].variableDomainsDict()

    # Create the new factor with the combined unconditioned and conditioned variables
    new_factor = Factor(unconditioned_vars, conditioned_vars, variable_domains_dict)

    # Step 3: Compute the probabilities
    for assignment in new_factor.getAllPossibleAssignmentDicts():
        probability = 1.0
        for factor in factors:
            probability *= factor.getProbability(assignment)
        new_factor.setProbability(assignment, probability)

    return new_factor

########### ########### ###########
########### QUESTION 3  ###########
########### ########### ###########

def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor: Factor, eliminationVariable: str):
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # Check that eliminationVariable is in unconditioned variables
        if eliminationVariable not in factor.unconditionedVariables():
            raise ValueError("Elimination variable must be an unconditioned variable in the factor.")

        # Check that factor has more than one unconditioned variable
        if len(factor.unconditionedVariables()) == 1:
            raise ValueError("Cannot eliminate the only unconditioned variable in the factor.")

        # Step 1: Define the new sets of variables
        new_unconditioned_vars = factor.unconditionedVariables() - {eliminationVariable}
        new_conditioned_vars = factor.conditionedVariables()

        # Step 2: Create a new factor with these variables
        new_factor = Factor(new_unconditioned_vars, new_conditioned_vars, factor.variableDomainsDict())

        # Step 3: Sum over the values of eliminationVariable
        for assignment in new_factor.getAllPossibleAssignmentDicts():
            probability_sum = 0.0
            # Iterate over all values of eliminationVariable
            for value in factor.variableDomainsDict()[eliminationVariable]:
                # Create a complete assignment including eliminationVariable
                complete_assignment = assignment.copy()
                complete_assignment[eliminationVariable] = value
                probability_sum += factor.getProbability(complete_assignment)
            # Set the computed sum into the new factor
            new_factor.setProbability(assignment, probability_sum)

        return new_factor

    return eliminate

eliminate = eliminateWithCallTracking()

