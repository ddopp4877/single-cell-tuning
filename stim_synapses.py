from allensdk.model.biophys_sim.config import Config
from allensdk.model.biophysical.utils import Utils
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

from allensdk.api.queries.biophysical_api import BiophysicalApi
from allensdk.model.biophys_sim.config import Config
from allensdk.model.biophysical.utils import Utils
from bmtool.synapses import SynapseTuner

from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from neuron import h
h.load_file("stdrun.hoc")  # Required to use h.run()

h.nrn_load_dll("modfiles/x86_64/.libs/libnrnmech.so")

import random
from neuron import h

class AllenCell:
    def __init__(self, gid, soma_diam_multiplier=1.0):
        
        self._gid = gid
        self.synapses = []  # Keep track of all synapses
        self.netcons = []   # Keep track of NetCons
        self.stims = []     # Keep VecStims so they don't get garbage-collected
        self.vecs = []
        self.syn_locs = [] 
        
        description = Config().load('manifest.json')

        self.utils = Utils(description)
        self.h = self.utils.h
        self.Vinit = self.utils.description.data['conditions'][0]['v_init']
        # Cast all genome values to float
        for d in self.utils.description.data['genome']:
            if 'value' in d:
                d['value'] = float(d['value'])

        # Load morphology and parameters
        morphology_path = description.manifest.get_path('MORPHOLOGY')
        self.utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))
        self.utils.load_cell_parameters()
        self.setup_morphology(soma_diam_multiplier)
        self._build_section_list()
    
    def setup_morphology(self,soma_diam_multiplier):

        self.soma = self.h.soma
        self.h.soma[0].diam *= soma_diam_multiplier
        self.dend = list(self.h.dend) if hasattr(self.h, 'dend') else []
        self.apic = list(self.h.apic) if hasattr(self.h, 'apic') else []
        self.axon = list(self.h.axon) if hasattr(self.h, 'axon') else []


    def _build_section_list(self):
        self.all = h.SectionList()
        for sec in h.allsec():
            self.all.append(sec)
    
    def add_random_synapses_from_df(self, df, syn_name, syn_params, section_pool=None):
        """
        Adds synapses at random locations with input times from a DataFrame.

        Parameters:
        - df: DataFrame with 'node' and 'timestamps' columns
        - syn_name: string name of NMODL synapse mechanism
        - syn_params: dictionary of synapse parameters (e.g. gmax, tau, e)
        - section_pool: list of sections to randomly choose from (default = self.dend)
        """
        if section_pool is None:
            section_pool = self.dend  # Default to dendrites

        for _, row in df.iterrows():
            timestamps = row['timestamps']
            if not timestamps:
                continue

            sec = random.choice(section_pool)
            loc = random.uniform(0.1, 0.9)
            self.syn_locs.append((sec, loc))
            
            # Create synapse
            syn = getattr(h, syn_name)(sec(loc))
            for param, val in syn_params.items():
                if hasattr(syn, param):
                    setattr(syn, param, val)

            # Create VecStim with spike times
            vec = h.Vector(timestamps)
            self.vecs.append(vec)
            stim = h.VecStim()
            stim.play(vec)
            self.stims.append(stim)

            nc = h.NetCon(stim, syn)
            nc.weight[0] = syn_params.get('weight', 0.001)

            self.synapses.append(syn)
            self.netcons.append(nc)


            
    def __str__(self):
        return f"AllenCell(soma={self.soma}, dendrites={len(self.dend)})"
    
cell = AllenCell(0)

syn_params =  {
            'initW': 3,
            'tau_r_AMPA': 5,
            'tau_d_AMPA': 4,
            'Use': 0.13,
            'Dep': 0.,
            'Fac': 200.
        }

PFR = pd.read_csv(os.path.join("..","pyrFiringRateAvg.csv"),delimiter=",")

PFR_time_shortened = np.array(PFR['Time'][PFR['Time'] >0])
PFR_firing_rate_shortened = np.array(PFR['AvgFiringRate'][PFR['Time'] >0])

num_neurons = 10000
psg = PoissonSpikeGenerator(population='example_pop', seed=32)

for i in range(num_neurons):
    psg.add(
        node_ids=i,
        firing_rate=PFR_firing_rate_shortened,
        times=PFR_time_shortened
    )

spikes_df = psg.to_dataframe()
print(f"Generated {len(spikes_df)} spikes across {num_neurons} neurons")

firing_rates = []
# Now loop over actual neuron IDs
for node_id in range(num_neurons):
    cell_spikes = spikes_df[spikes_df['node_ids'] == node_id]
    duration = PFR_time_shortened[-1] - PFR_time_shortened[0]
    firing_rate = len(cell_spikes) / duration
    firing_rates.append(firing_rate)
    
N = 1000  
df_sampled = spikes_df.sample(n=N, random_state=42)

cell.add_random_synapses_from_df(df_sampled, "AMPA_NMDA_STP", syn_params, section_pool=cell.dend)


print(f"Added {len(cell.synapses)} synapses")
for i, syn in enumerate(cell.synapses):
    print(f"Synapse {i}: {syn.get_segment().sec.name()} @ {syn.get_segment().x}")
    


# Set recording vectors
tvec = h.Vector().record(h._ref_t)
vvec = h.Vector().record(cell.soma[0](0.5)._ref_v)  # somatic Vm

# Simulation control
h.finitialize(cell.Vinit)
h.tstop = 100.0  # ms
h.run()


import matplotlib.pyplot as plt

# Collect synapse coordinates
x_pts, y_pts = [], []

for sec, loc in cell.syn_locs:
    # Interpolate pt3d coordinates at the synapse location
    n3d = int(h.n3d(sec=sec))
    if n3d < 2:
        continue  # skip sections with too little data
    arc_lengths = [h.arc3d(i, sec=sec) for i in range(n3d)]
    total_length = arc_lengths[-1]
    target = loc * total_length

    # Find the pt3d segment closest to the target length
    for i in range(n3d - 1):
        if arc_lengths[i] <= target <= arc_lengths[i+1]:
            frac = (target - arc_lengths[i]) / (arc_lengths[i+1] - arc_lengths[i])
            x0, x1 = h.x3d(i, sec=sec), h.x3d(i+1, sec=sec)
            y0, y1 = h.y3d(i, sec=sec), h.y3d(i+1, sec=sec)
            x_pts.append(x0 + frac * (x1 - x0))
            y_pts.append(y0 + frac * (y1 - y0))
            break

# Plot
plt.figure()
plt.scatter(x_pts, y_pts, color='red', s=10, label='Synapses')
plt.axis('equal')
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.title("Synapse Locations on Morphology")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("synapse_locations.png")
plt.show()


plt.figure()
plt.plot(np.array(tvec),np.array(vvec))
plt.savefig("output.png")