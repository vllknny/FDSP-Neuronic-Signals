import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

start_scope()



# The weights are just interest_i × w_max, so higher-interest stakeholders push the neuron toward threshold harder. 
# Multiplying by N keeps the average weight at w_max after normalizing.
# Vested interest here seems arbitrary as implementation has no real calculation behind it. Next step.
# --- Vested interest -> synaptic weights ---
interests = np.array([0.9, 0.7, 0.5, 0.3, 0.1])   # one per stakeholder
interests = interests / interests.sum()            # normalize (optional)
N = len(interests)
w_max = 8*mV                                       # weight at full interest

# --- The single neuron ---
tau, v_rest, v_th, v_reset = 20*ms, -70*mV, -50*mV, -70*mV
eqs = 'dv/dt = (v_rest - v)/tau : volt (unless refractory)'
neuron = NeuronGroup(1, eqs, threshold='v > v_th', reset='v = v_reset',
                     refractory=5*ms, method='exact')
neuron.v = v_rest

# --- Stakeholder inputs (Poisson spike trains) ---
stakeholders = PoissonGroup(N, rates=40*Hz)

# --- Synapses: weight proportional to vested interest ---
S = Synapses(stakeholders, neuron, 'w : volt', on_pre='v_post += w')
S.connect()                                        # all N -> the one neuron
S.w = 'interests[i] * w_max * N'                   # i = presynaptic index

# --- Monitors ---
spikes_in = SpikeMonitor(stakeholders)
spikes_out = SpikeMonitor(neuron)
vm = StateMonitor(neuron, 'v', record=0)

run(500*ms)

# --- Plot ---
fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
ax[0].plot(spikes_in.t/ms, spikes_in.i, '.k')
ax[0].set_ylabel('stakeholder')
ax[1].plot(vm.t/ms, vm.v[0]/mV)
ax[1].set_ylabel('v (mV)')
ax[2].plot(spikes_out.t/ms, np.zeros(len(spikes_out.t)), '|r', ms=20)
ax[2].set_ylabel('output'); ax[2].set_xlabel('time (ms)')
plt.tight_layout(); plt.show()

print("Weights (mV):", S.w[:] / mV)
print("Output spikes:", spikes_out.num_spikes)