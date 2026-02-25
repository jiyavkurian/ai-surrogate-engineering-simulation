# READ RESULTS FROM ODB FILES

from odbAccess import *
import os

N_samples = 50

print('Extracting results from %d samples...\n' % N_samples)

max_mises_results = []

for k in range(1, N_samples + 1):
    odb_name = 'RotatingDisk_Sample_%d.odb' % k
    
    if not os.path.exists(odb_name):
        print('Sample %d: File not found' % k)
        max_mises_results.append(None)
        continue
    
    try:
        odb = openOdb(path=odb_name, readOnly=True)
        
        # Get step (try common names)
        if 'LoadStep' in odb.steps:
            step = odb.steps['LoadStep']
        else:
            step = odb.steps[odb.steps.keys()[0]]
        
        # Get max von Mises stress
        stress = step.frames[-1].fieldOutputs['S']
        max_mises = max(v.mises for v in stress.values if hasattr(v, 'mises') and v.mises)
        
        max_mises_results.append(max_mises)
        print('Sample %d: %.2f MPa' % (k, max_mises/1e6))
        odb.close()
        
    except Exception as e:
        print('Sample %d: ERROR - %s' % (k, str(e)))
        max_mises_results.append(None)

# Print summary
valid = [r for r in max_mises_results if r is not None]
print('\n' + '='*50)
print('Successful: %d/%d' % (len(valid), N_samples))
if valid:
    print('Mean stress: %.2f MPa' % (sum(valid)/len(valid)/1e6))
    print('Range: %.2f - %.2f MPa' % (min(valid)/1e6, max(valid)/1e6))

# Write CSV
with open('ABAQUS_stress_results.csv', 'w') as f:
    f.write('Sample,max_mises_Pa\n')
    for i, r in enumerate(max_mises_results):
        f.write('%d,%s\n' % (i+1, '%.6e' % r if r else 'NaN'))

# Write combined results (inputs + outputs)
with open('uq_combined_results_Al.csv', 'w') as f:
    f.write('Sample,E_Pa,rho_kg_m3,omega_rad_s,sigma_uts_Pa,max_mises_Pa\n')
    for k in range(1, N_samples + 1):
        inp_file = 'Inputs_Sample_%d.txt' % k
        if os.path.exists(inp_file):
            with open(inp_file, 'r') as inp:
                lines = inp.readlines()
                E, rho, omega, sigma_uts = [float(lines[i]) for i in range(4)]
            r = max_mises_results[k-1]
            f.write('%d,%.6e,%.4f,%.6f,%.6e,%s\n' % 
                    (k, E, rho, omega, sigma_uts, '%.6e' % r if r else 'NaN'))

print('\nResults saved to: ABAQUS_stress_results.csv, uq_combined_results.csv')
