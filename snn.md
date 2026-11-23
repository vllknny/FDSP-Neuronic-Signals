Documentation of the snn as global scope will go here, as explanation within the Python File is needlessly verbose and complicated.

We have the SNN main variables:
Global Scope:
    interests == weights within an array
    neuron
    synapse via interest
    spike via interest
        eq
    stakeholder input via Poisson Distr.

Interest is the highest determining variable, as vested interest is our main form of representation of each human within the signal.

Determining what interest should be is the next step.

-- Vested interest should not be confused with vested in interest. This term, unlike vested interest, applies to entities such as trusts. The beneficiary of a trust is vested in interest if they do not have to meet any condition for their interest to take effect. In this case, the recipient has a present right to future enjoyment, such as a right to property when another beneficiary's interest ends. In this case, that beneficiary has access to the property when the primary beneficiary becomes deceased. --
Pasted from Investopedia (https://www.investopedia.com/terms/v/vestedinterest.asp)

I found a few notes within this article to be a good way to finding determinants.
Some worthy variables within implementation:

-- current stake (interest already put in, series calc)
-- influence factor (word of mouth, news, trendline, previous macroanalysis in investments of curr and previous portfolio)
-- salience, what is the beneficial yield. "...prominence of an issue, as perceived by an individual, shapes the strength of his resulting attitude"
-- certainty, what is the distribution of the yield graph.
-- immediacy, time of possible investment producing positive yield. (i.e. long term/short term stock, need based)
-- the hardest calculation, efficacy, or capability of snn net. this is based off of essentially a confidence interval of a human.

The most interesting part of these variables is the link within communication theory, and current data fields that inherently produce data representing
this factors. AdSense, marketing, market history, GSearch API, and others can be relatively reverse engineered to create more accurate vested interest
weights. This is by far, the most crucial part of the EDA behind both finance and quantitative domains.

This will be the next step I will work on.

Variable |	        Implementation |	    Role
Current stake	    stake	Base term.      Sunk interest sets the floor.
Influence factor	1 + β·influence	        Multiplier for reach (news, word of mouth, trend, portfolio history).
Salience	        |y_mu| normalized	    Magnitude of the perceived yield.
Certainty	        1/(1+CV²)	            Tight yield distribution gives a weight near 1, a wide one shrinks it.
Immediacy	        exp(-δt)	            Exponential discounting of time to yield.
Efficacy	        bounded 0..1	        Confidence that the player can move the outcome.

The weights feed the game in two places. The share ω_i of total vested interest sets both the synaptic weight and how much each player counts toward coalition strength C. Each player's payoff then depends on C, and a quantal-response fixed point gives their action probability, which becomes their Poisson rate. Opponents (direction = -1) get negative weights, so a high-interest regulator acts as a veto on the neuron.

Caveats on Accuracy

The multiplicative form and the exponents are modeling choices, not derived results. For accuracy, treat them as free parameters (beta, delta, or per-term exponents) and fit them against observed outcomes such as who actually committed capital and when.
Efficacy is the weakest input here, I left it as a bounded 0..1 input. Track record, for example a Beta posterior over past successful interventions, taking its mean as efficacy and its width as a penalty would be ideal.
Data sources would replace the placeholder arrays. Search or ad-spend intensity could feed influence and salience, market history could feed y_mu/y_sd, and position or filing data could feed stake. Each is a proxy we should find how validation factors behavior.