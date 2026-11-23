import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

start_scope()

# ============================================================
# 1. PLAYERS & RAW INPUTS (replace with real data later)
# ============================================================
players   = ['Whale', 'Institution', 'Retail', 'Regulator', 'Insider']
N = len(players)

stake     = np.array([0.90, 0.70, 0.20, 0.10, 0.60])   # capital/effort already committed
influence = np.array([0.80, 0.90, 0.30, 0.95, 0.50])   # reach: news, word of mouth, trend, portfolio history
y_mu      = np.array([1.00, 0.80, 0.40, 0.60, 0.90])   # expected yield magnitude (salience driver)
y_sd      = np.array([0.30, 0.20, 0.60, 0.15, 0.50])   # yield dispersion (certainty driver)
t_payoff  = np.array([2.0,  1.0,  0.25, 3.0,  0.5])    # years until yield (immediacy driver)
efficacy  = np.array([0.90, 0.85, 0.30, 0.70, 0.60])   # confidence in ability to move the outcome, 0..1
direction = np.array([+1,   +1,   +1,   -1,   +1])    # +1 supports the action, -1 opposes (veto/blocker)
cost      = np.array([0.30, 0.25, 0.10, 0.20, 0.25])   # cost of participating in the game

# ============================================================
# 2. VESTED INTEREST -> WEIGHTS
# ============================================================
def vested_interest(stake, influence, y_mu, y_sd, t, efficacy,
                    beta=1.0, delta=0.5):
    stake_t    = stake                                     # current stake (sunk interest)
    influence_t = 1 + beta * influence                      # amplifier, 1..1+beta
    salience   = np.abs(y_mu) / np.abs(y_mu).max()          # prominence/magnitude of yield
    certainty  = 1 / (1 + (y_sd / np.abs(y_mu))**2)         # 1/(1+CV^2): tight yield dist -> ~1
    immediacy  = np.exp(-delta * t)                         # discounting of time to yield
    return stake_t * influence_t * salience * certainty * immediacy * efficacy

raw    = vested_interest(stake, influence, y_mu, y_sd, t_payoff, efficacy)
omega  = raw / raw.sum()                                    # share of total vested interest
w_mV   = direction * omega * N * 6        # signed; mean |w| = 6 mV. negative = inhibitory

# ============================================================
# 3. GAME: quantal-response equilibrium over participation
# ============================================================
def sigmoid(x): return 1 / (1 + np.exp(-x))

def solve_game(omega, direction, y_mu, cost, lam=6.0, iters=500, damp=0.9):
    a = np.full(len(omega), 0.5)                            # P(player acts)
    for _ in range(iters):
        # Coalition success = interest-weighted net participation, clipped to [0,1]
        C = np.clip(np.sum(direction * omega * a) / omega.sum() * 2, 0, 1)
        # Supporters earn yield if coalition succeeds; opponents earn it if it fails
        EU = np.where(direction > 0, y_mu * C, y_mu * (1 - C)) - cost
        a  = damp * a + (1 - damp) * sigmoid(lam * EU)
    return a, C

a_eq, C_eq = solve_game(omega, direction, y_mu, cost)
rates = a_eq * 60 * Hz                                      # equilibrium action prob -> firing rate

# ============================================================
# 4. SNN
# ============================================================
tau, v_rest, v_th, v_reset = 20*ms, -70*mV, -50*mV, -70*mV
neuron = NeuronGroup(1, 'dv/dt = (v_rest - v)/tau : volt (unless refractory)',
                     threshold='v > v_th', reset='v = v_reset',
                     refractory=5*ms, method='exact')
neuron.v = v_rest

stakeholders = PoissonGroup(N, rates=rates)

S = Synapses(stakeholders, neuron, 'w : volt', on_pre='v_post += w')
S.connect()
S.w = w_mV * mV

spk_in  = SpikeMonitor(stakeholders)
spk_out = SpikeMonitor(neuron)
vm      = StateMonitor(neuron, 'v', record=0)

run(1000*ms)

# ============================================================
# 5. OUTPUT
# ============================================================
print(f"{'player':12s} {'omega':>6s} {'w(mV)':>7s} {'P(act)':>7s}")
for p, o, w, a in zip(players, omega, w_mV, a_eq):
    print(f"{p:12s} {o:6.3f} {w:7.2f} {a:7.2f}")
print(f"Coalition strength C = {C_eq:.2f} | decision spikes = {spk_out.num_spikes}")

fig, ax = plt.subplots(3, 1, figsize=(8, 8))
ax[0].bar(players, w_mV, color=['tab:blue' if w > 0 else 'tab:red' for w in w_mV])
ax[0].set_ylabel('weight (mV)')
ax[1].plot(spk_in.t/ms, spk_in.i, '.k'); ax[1].set_yticks(range(N)); ax[1].set_yticklabels(players)
ax[1].set_xlim(0, 1000)
ax[2].plot(vm.t/ms, vm.v[0]/mV); ax[2].axhline(-50, ls='--', c='r')
ax[2].set_ylabel('v (mV)'); ax[2].set_xlabel('time (ms)')
plt.tight_layout(); plt.show()